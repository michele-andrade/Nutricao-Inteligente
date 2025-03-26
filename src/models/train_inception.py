import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.transforms import ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim
import torch
from pathlib import Path
import matplotlib.pyplot as plt


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

def custom_collate_fn(batch):
    data = [item for item in batch if item[0] != 'error']
    if len(data) == 0: return torch.Tensor(), torch.Tensor()
    images, labels = zip(*data)
    return torch.stack(images), torch.stack(labels)


class ImageDataGenerator(Dataset):
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
                if img_file.is_file():
                    self.image_paths.append(str(img_file))
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



import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, base_model, num_tasks=5):
        super(MyModel, self).__init__()
        
        # Pegar a saída mixed5c. Em InceptionV3, isso é feito cortando o modelo antes do "Mixed_6a".
        modules = list(base_model.children())[:-3]  # -3 para parar no mixed5c
        self.features = nn.Sequential(*modules)
        
        # Average pooling 3x3, stride 2
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        
        # Duas camadas FC
        #self.fc1 = nn.Linear(2048, 4096)
        self.fc1 = nn.Linear(18432, 4096)

        self.fc2 = nn.Linear(4096, 4096)
        
        # Camadas finais para cada tarefa
        self.task_outputs = nn.ModuleList([nn.Sequential(nn.Linear(4096, 4096), nn.Linear(4096, 1)) for _ in range(num_tasks)])

    def forward(self, x):
        #print("Shape of x before features:", x.shape)
        x = self.features(x)
        #print(self.features[-1])
        #print("Shape after self.features:", x.shape)
        x = self.avgpool(x)
        #print("Shape after avgpool:", x.shape)

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        #print("Shape after flattening:", x.shape)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Task-specific outputs
        outputs = [task(x) for task in self.task_outputs]
        
        return torch.cat(outputs, dim=1)

def build_model():
    #base_model = models.inception_v3(pretrained=True)
    base_model = models.inception_v3(pretrained=True, aux_logits=False)

    for param in base_model.parameters():
        param.requires_grad = False
    
    return MyModel(base_model)



root_path = '/media/work/datasets/nutrition5k_dataset/imagery/side_angles'
dish_metadata_cafe1 = process_dish_metadata('/media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv')
dish_metadata_cafe2 = process_dish_metadata('/media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv')
dish_metadata = pd.concat([dish_metadata_cafe1, dish_metadata_cafe2], ignore_index=True)

train_ids = load_split_ids('/media/work/datasets/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt')
test_ids = load_split_ids('/media/work/datasets/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt')

BATCH_SIZE = 16
EPOCHS = 15

train_dataset = ImageDataGenerator(root_path, dish_metadata, train_ids, target_size=(299, 299))
test_dataset = ImageDataGenerator(root_path, dish_metadata, test_ids, target_size=(299, 299))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

sample_batch = next(iter(train_loader))
print(sample_batch[0].shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_model().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_losses = []
val_losses = []
best_val_loss = float('inf')

# No seu loop de treinamento...
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
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            if len(inputs) == 0 or len(targets) == 0:  # Verificando se o lote é vazio.
                print("Batch vazio detectado no loop de teste!")
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    val_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}, Validation Loss: {epoch_loss}')

    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        torch.save(model.state_dict(), 'output/best_inception_model_new.pth')

torch.save(model.state_dict(), 'output/inception_model_final_new.pth')

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('output/loss_curve_incpetion_new.png')