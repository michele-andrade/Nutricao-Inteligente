# generate_predictions_ViT_imagenet.py

import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from PIL import Image

################################################################################
# Utility Functions
################################################################################

def custom_collate_fn(batch):
    """
    Custom collate function to filtrar amostras inválidas (None).
    Se o batch inteiro for inválido, retorna None.
    
    Args:
        batch (list): Lista de tuplas (image_tensor, target_tensor, dish_id).
    
    Returns:
        (Tensor, Tensor, list) ou None
    """
    valid_samples = [item for item in batch if item is not None]
    if not valid_samples:
        return None
    images, targets, dish_ids = zip(*valid_samples)
    return torch.stack(images), torch.stack(targets), list(dish_ids)


def process_dish_metadata(file_path):
    """
    Lê o CSV de metadados de Nutrition5k e retorna um DataFrame com colunas:
      ['dish_id','total_calories','total_mass','total_fat','total_carb','total_protein'].
    
    Args:
        file_path (str): Caminho do CSV.
    
    Returns:
        pd.DataFrame
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[1:]:  # pula o cabeçalho
        fields = line.strip().split(',')
        if len(fields) < 6:
            continue  # Linha malformada, pula
        dish_id       = fields[0]
        total_calories= float(fields[1])
        total_mass    = float(fields[2])
        total_fat     = float(fields[3])
        total_carb    = float(fields[4])
        total_protein = float(fields[5])
        data.append((dish_id, total_calories, total_mass, total_fat, total_carb, total_protein))

    return pd.DataFrame(data, columns=[
        'dish_id','total_calories','total_mass','total_fat','total_carb','total_protein'
    ])


################################################################################
# Dataset Class para Inference
################################################################################

class Nutrition5kInferenceDataset(Dataset):
    """
    Carrega imagens de Nutrition5k para inferência.
    Retorna (image_tensor, target_tensor_dummy, dish_id).
    """
    def __init__(self, 
                 root_path, 
                 metadata_df, 
                 dish_ids, 
                 target_size=(224, 224)):
        """
        Args:
            root_path (str): Caminho base onde ficam subpastas com dish_id/frames_sampled.
            metadata_df (pd.DataFrame): Metadados com colunas
                ['dish_id','total_calories','total_mass','total_fat','total_carb','total_protein'].
            dish_ids (List[str]): Lista de dish_ids para inferência.
            target_size (tuple): Tamanho para resize (largura, altura).
        """
        super().__init__()
        self.root_path   = Path(root_path)
        self.metadata_df = metadata_df
        self.target_size = target_size

        # Dicionário dish_id -> [calorias, mass, fat, carb, protein]
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
            # Ajuste se o seu subdiretório não for 'frames_sampled10'
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

        # Carrega imagem com PIL ou OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Falha ao ler: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        # Converte para tensor
        img_tensor = ToTensor()(img)
        # Normalização padrão do ImageNet
        img_tensor = Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img_tensor)

        # Opcional: resgatar target real do metadata. Para inferência pura, não é necessário.
        if dish_id in self.metadata_dict:
            target_vals = self.metadata_dict[dish_id]
            target_tensor = torch.tensor(target_vals, dtype=torch.float32)
        else:
            target_tensor = torch.zeros(5, dtype=torch.float32)

        return img_tensor, target_tensor, dish_id


################################################################################
# Modelo ViT de Regressão (mesmo da fase de treinamento)
################################################################################
import torch.nn as nn
from transformers import ViTModel

class ViTRegressionModel(nn.Module):
    """
    ViT base com cabeça de regressão para 5 valores (cal, mass, fat, carb, protein).
    Deve ser o MESMO que você usou no treino!
    """
    def __init__(self, pretrained_name="google/vit-base-patch16-224", output_dim=5):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.vit.config.hidden_size  # geralmente 768 para base
        self.regression_head = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        x: Tensor de imagens no formato [batch, 3, 224, 224]
        Return: Tensor [batch, 5]
        """
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # pega o [CLS]
        return self.regression_head(cls_token)


################################################################################
# Função principal para gerar predições
################################################################################

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
    Gera predições usando um modelo ViTRegressionModel (PyTorch) para Nutrition5k.

    Args:
        model (nn.Module): Instância do ViTRegressionModel (5 saídas).
        model_weights_path (str): Caminho para o arquivo .pth com pesos treinados.
        root_path (str): Caminho base para side_angles/<dish_id>/frames_sampled10
        metadata_paths (List[str]): CSVs de metadados (cafe1, cafe2, etc.)
        test_ids_path (str): TXT com dish_ids para inferência.
        output_csv_path (str): Caminho do CSV final de predições.
        batch_size (int): Tamanho do batch no DataLoader. Default 8.
        target_size (tuple): Resize das imagens. Default (224, 224).
        num_workers (int): Workers do DataLoader. Default 4.
        device (str|torch.device): CPU ou GPU. Se None, detecta automaticamente.

    Output CSV: colunas [dish_id, pred_calories, pred_mass, pred_fat, pred_carb, pred_protein]
    """
    # 1) Dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Carregar pesos
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 3) Carregar metadados
    dfs = []
    for mpath in metadata_paths:
        df_tmp = process_dish_metadata(mpath)
        dfs.append(df_tmp)
    if len(dfs) == 0:
        raise ValueError("Nenhum CSV de metadata fornecido!")
    combined_metadata = pd.concat(dfs, ignore_index=True)

    # 4) IDs de teste
    with open(test_ids_path, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]

    # 5) Dataset e DataLoader
    dataset = Nutrition5kInferenceDataset(
        root_path=root_path,
        metadata_df=combined_metadata,
        dish_ids=test_ids,
        target_size=target_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 6) Inferência e escrita no CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dish_id","pred_calories","pred_mass","pred_fat","pred_carb","pred_protein"])

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                images, _, dish_ids = batch
                images = images.to(device)

                preds = model(images)  # [batch, 5]
                preds = preds.cpu().numpy()

                for dish_id, pred_row in zip(dish_ids, preds):
                    writer.writerow([dish_id] + pred_row.tolist())

    print(f"[INFO] Predições salvas em: {output_csv_path}")


################################################################################
# Exemplo de uso
################################################################################

if __name__ == "__main__":
    """
    Para rodar no terminal:
        python3 generate_predictions_ViT_imagenet.py

    Ajuste os caminhos abaixo conforme seu ambiente e nomes de arquivos.
    """
    # 1) Instancia o MESMO modelo que foi treinado
    model_example = ViTRegressionModel(
        pretrained_name="google/vit-base-patch16-224",  # ID Hugging Face usado no treino
        output_dim=5
    )

    # 2) Caminhos
    model_weights = "output/vit_imagenet_best.pth"  # pesos salvos ao final do treino
    root = "data/nutrition5k_dataset/imagery/side_angles"
    metadata_files = [
        "data/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
        "data/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv"
    ]
    test_ids_file = "data/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt"
    output_csv = "output/predictions_vit_imagenet.csv"

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

    print("Done. Verifique o arquivo CSV de saída.")

