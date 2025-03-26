###############################################################################
# generate_predictions_CLIP_coyo.py
###############################################################################
import os
import csv
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from transformers import CLIPModel

###############################################################################
# 1) Dataset e Utilitários
###############################################################################
def custom_collate_fn(batch):
    """
    Filtra amostras inválidas (None). Retorna None se batch inteiro estiver vazio.
    
    batch: lista de (img_tensor, dummy_target, dish_id).
    """
    valid_samples = [item for item in batch if item is not None]
    if not valid_samples:
        return None
    images, targets, dish_ids = zip(*valid_samples)
    return torch.stack(images), torch.stack(targets), list(dish_ids)


def process_dish_metadata(file_path):
    """
    Lê CSV de metadados para Nutrition5k:
      dish_id, total_calories, total_mass, total_fat, total_carb, total_protein
    Retorna DataFrame com essas colunas.
    """
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # pula cabeçalho
        fields = line.strip().split(',')
        if len(fields) < 6:
            continue
        dish_id       = fields[0]
        total_calories= float(fields[1])
        total_mass    = float(fields[2])
        total_fat     = float(fields[3])
        total_carb    = float(fields[4])
        total_protein = float(fields[5])
        data.append((dish_id, total_calories, total_mass, total_fat, total_carb, total_protein))

    return pd.DataFrame(data, columns=[
        'dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein'
    ])


class Nutrition5kInferenceDataset(Dataset):
    """
    Dataset para inferência, retorna (image_tensor, dummy_target, dish_id).
    """
    def __init__(self, root_path, metadata_df, dish_ids, target_size=(224, 224)):
        """
        root_path (str): caminho p/ side_angles/<dish_id>/frames_sampled10
        metadata_df (pd.DataFrame)
        dish_ids (List[str]): IDs para inferência
        target_size (tuple): resize (224,224) padrão
        """
        super().__init__()
        self.root_path = Path(root_path)
        self.metadata_df = metadata_df
        self.target_size = target_size

        # Cria dicionário dish_id -> [cal, mass, fat, carb, protein]
        self.metadata_dict = {}
        for row in metadata_df.itertuples(index=False):
            self.metadata_dict[row.dish_id] = [
                row.total_calories, 
                row.total_mass, 
                row.total_fat, 
                row.total_carb, 
                row.total_protein
            ]

        self.image_paths = []
        self.dish_labels = []
        for dish_id in dish_ids:
            frames_dir = self.root_path / dish_id / 'frames_sampled10'
            if not frames_dir.is_dir():
                continue

            for img_file in frames_dir.iterdir():
                if img_file.is_file():
                    self.image_paths.append(str(img_file))
                    self.dish_labels.append(dish_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        dish_id  = self.dish_labels[idx]

        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Não foi possível ler a imagem em: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        img_tensor = ToTensor()(img)
        img_tensor = Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])(img_tensor)

        # Não precisamos do target real para inferência pura. Usamos zeros ou o real do dict.
        if dish_id in self.metadata_dict:
            target_vals = self.metadata_dict[dish_id]
            target_tensor = torch.tensor(target_vals, dtype=torch.float32)
        else:
            target_tensor = torch.zeros(5, dtype=torch.float32)

        return img_tensor, target_tensor, dish_id


###############################################################################
# 2) CLIP ViT Regression Model (MESMO da Fase de Treino)
###############################################################################
class CLIPViTRegressionModel(nn.Module):
    """
    CLIP's Vision Transformer com cabeça de regressão para 5 valores.
    Foi como no seu script de treino, mas adaptado à inference.
    """
    def __init__(self, pretrained_name="openai/clip-vit-base-patch32", output_dim=5):
        super(CLIPViTRegressionModel, self).__init__()
        clip_model = CLIPModel.from_pretrained(pretrained_name)
        self.vit = clip_model.vision_model
        hidden_size = self.vit.config.hidden_size  # normalmente 768
        self.regression_head = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Recebe tensor [batch, 3, H, W], retorna [batch, 5].
        """
        outputs = self.vit(x)
        # outputs.last_hidden_state -> [batch, seq_len, hidden_size]
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(cls_token)


###############################################################################
# 3) Função principal para gerar predições
###############################################################################
def generate_predictions(model,
                         model_weights_path,
                         root_path,
                         metadata_paths,
                         test_ids_path,
                         output_csv_path,
                         batch_size=8,
                         target_size=(224, 224),
                         num_workers=4,
                         device=None):
    """
    Gera predições usando CLIPViTRegressionModel em dados do Nutrition5k.
    
    Args:
        model (nn.Module): Instância de CLIPViTRegressionModel
        model_weights_path (str): Caminho .pth com pesos treinados
        root_path (str): pasta base dos frames (side_angles)
        metadata_paths (List[str]): CSVs com metadados
        test_ids_path (str): .txt com dish_ids de inferência
        output_csv_path (str): arquivo final de predições
        batch_size (int): batch size (default=8)
        target_size (tuple): resize das imagens (default=224x224)
        num_workers (int): workers no DataLoader
        device: CPU ou GPU (torch.device). Se None, autodetect.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega pesos
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Concatena metadados
    df_list = []
    for mpath in metadata_paths:
        df_list.append(process_dish_metadata(mpath))
    if not df_list:
        raise ValueError("Nenhum CSV de metadado informado!")
    combined_metadata = pd.concat(df_list, ignore_index=True)

    # Carrega IDs de teste
    with open(test_ids_path, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]

    # Cria Dataset & DataLoader
    test_dataset = Nutrition5kInferenceDataset(
        root_path=root_path,
        metadata_df=combined_metadata,
        dish_ids=test_ids,
        target_size=target_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Inferência
    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dish_id","pred_calories","pred_mass","pred_fat","pred_carb","pred_protein"])
        
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                images, _, dish_ids = batch
                images = images.to(device)

                preds = model(images)   # [batch, 5]
                preds = preds.cpu().numpy()

                for dish_id, row_preds in zip(dish_ids, preds):
                    writer.writerow([dish_id] + row_preds.tolist())

    print(f"[INFO] Predições salvas em: {output_csv_path}")


###############################################################################
# 4) Exemplo de uso
###############################################################################
if __name__ == "__main__":
    """
    Para rodar no terminal (ajuste caminhos conforme seu setup):
      python3 generate_predictions_CLIP_coyo.py
    """
    # 1) Carregue a MESMA arquitetura usada no treino
    #    (No script de treino, instanciou: CLIPViTRegressionModel('openai/clip-vit-base-patch32', 5))
    model_example = CLIPViTRegressionModel(
        pretrained_name="openai/clip-vit-base-patch32",
        output_dim=5
    )

    # 2) Caminhos
    model_weights = "output/vit_coyo_best.pth"  # Pesos do modelo treinado
    root = "data/nutrition5k_dataset/imagery/side_angles"
    metadata_files = [
        "data/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
        "data/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv"
    ]
    test_ids_file = "data/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt"
    output_csv = "output/predictions_vit_COYO.csv"

    # 3) Gerar predições
    generate_predictions(
        model=model_example,
        model_weights_path=model_weights,
        root_path=root,
        metadata_paths=metadata_files,
        test_ids_path=test_ids_file,
        output_csv_path=output_csv,
        batch_size=8,
        target_size=(224, 224)
    )

    print("[INFO] Done! Verifique o arquivo de saída:", output_csv)
