import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import Resize
from torch import nn, optim
import torch
from torchvision.models import resnet50
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.transforms import ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    data = [item for item in batch if item[0] != 'error']
    if len(data) == 0: return torch.Tensor(), torch.Tensor()
    images, labels = zip(*data)
    return torch.stack(images), torch.stack(labels)


def process_dish_metadata(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[1:]:
        fields = line.split(',')
        dish_id = fields[0]
        total_calories = float(fields[1])
        total_mass = float(fields[2])
        total_fat = float(fields[3])
        total_carb = float(fields[4])
        total_protein = float(fields[5])
        data.append((dish_id, total_calories, total_mass, total_fat, total_carb, total_protein))

    return pd.DataFrame(data, columns=['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein'])

def load_split_ids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

class ImageDataGenerator(Dataset):
    # def __init__(self, root_path, metadata_df, split_ids, target_size=(256, 256), shuffle=True):
    #     self.root_path = root_path
    #     self.metadata_df = metadata_df
    #     self.target_size = target_size
    #     self.shuffle = shuffle

    #     self.image_paths = []
    #     self.image_labels = []
    #     for dish_id in split_ids:
    #         dish_dir = os.path.join(self.root_path, dish_id, 'frames_sampled5')
    #         for img_file in os.listdir(dish_dir):
    #             img_path = os.path.join(dish_dir, img_file)
    #             self.image_paths.append(img_path)
    #             self.image_labels.append(dish_id)

    #     if self.shuffle:
    #         temp = list(zip(self.image_paths, self.image_labels))
    #         np.random.shuffle(temp)
    #         self.image_paths, self.image_labels = zip(*temp)

    def __init__(self, root_path, metadata_df, split_ids, target_size=(256, 256), shuffle=True):
        self.root_path = root_path
        self.metadata_df = metadata_df
        self.target_size = target_size
        self.shuffle = shuffle

        self.image_paths = []
        self.image_labels = []
        for dish_id in split_ids:
            dish_dir = Path(self.root_path) / dish_id / 'frames_sampled5'
            for img_file in dish_dir.iterdir():
                if img_file.is_file():  # Only consider files, not sub-directories
                    img_path = str(img_file)
                    self.image_paths.append(img_path)
                    self.image_labels.append(dish_id)

        if self.shuffle:
            temp = list(zip(self.image_paths, self.image_labels))
            np.random.shuffle(temp)
            self.image_paths, self.image_labels = zip(*temp)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        dish_id = self.image_labels[index]

        # if not os.path.exists(img_path):
        #     return None, None
        if not os.path.exists(img_path):
            return 'error', 'error'

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image at {img_path}")
            return 'error', 'error'
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, self.target_size)
        img = ToTensor()(img)
        img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) 

        metadata = self.metadata_df[self.metadata_df['dish_id'] == dish_id]
        if metadata.empty: 
            print(f"No metadata found for dish_id {dish_id}")
            return img, torch.zeros(5)
        target = metadata[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']].values[0]

        return img, torch.tensor(target, dtype=torch.float32)



class MyModel(nn.Module):
    def __init__(self, base_model, outputs):
        super(MyModel, self).__init__()
        self.base_model = base_model
        self.outputs = outputs

    def forward(self, x):
        x = self.base_model(x)
        x = [output(x) for output in self.outputs]
        return torch.cat(x, dim=1)

def build_model():
    base_model = resnet50(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False

    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, 4096)

    outputs = nn.ModuleList([
        nn.Linear(4096, 1) for _ in range(5)
    ])

    return MyModel(base_model, outputs)


root_path = '/media/work/datasets/nutrition5k_dataset/imagery/side_angles'
dish_metadata_cafe1 = process_dish_metadata('/media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv')
dish_metadata_cafe2 = process_dish_metadata('/media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv')
dish_metadata = pd.concat([dish_metadata_cafe1, dish_metadata_cafe2], ignore_index=True)

train_ids = load_split_ids('/media/work/datasets/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt')
test_ids = load_split_ids('/media/work/datasets/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt')

BATCH_SIZE = 8
EPOCHS = 60

train_dataset = ImageDataGenerator(root_path, dish_metadata, train_ids)
test_dataset = ImageDataGenerator(root_path, dish_metadata, test_ids)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_model().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
# criterion = nn.L1Loss()
criterion = nn.MSELoss() # *** teste MSE como loss!

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # gamma é o fator pelo qual o lr será multiplicado.

train_losses = []
val_losses = []
best_val_loss = float('inf')  # Inicia a melhor perda de validação como infinito.


# loop de treinamento...
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        if batch is None:
            continue

        inputs, targets = zip(*[(img, target) for img, target in zip(*batch) if img is not None and target is not None])
        inputs = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    scheduler.step()  # Atualiza a taxa de aprendizado


    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)  # Atualiza a lista de perdas de treinamento.
    print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

    model.eval()
    running_loss = 0.0
    # Testing
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            if inputs is None or targets is None:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    val_losses.append(epoch_loss)  # Atualiza a lista de perdas de validação.
    print(f'Epoch {epoch+1}, Validation Loss: {epoch_loss}')

    # Salva o melhor modelo.
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        torch.save(model.state_dict(), 'output/best_model_resnet_MSE.pth')

# Save the trained model
torch.save(model.state_dict(), 'output/resnet50_model_final_MSE.pth')

# Plot the training and validation loss curves.
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('output/loss_curve_resnet_MSE_loss.png')  # Salva o gráfico como uma imagem.
