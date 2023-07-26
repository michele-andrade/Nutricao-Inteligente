import torch
from torchvision.transforms import ToTensor, Normalize
import cv2
import numpy as np
from torchvision.models import resnet50
from torch import nn

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
    base_model = resnet50(pretrained=False)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, 1024)

    outputs = nn.ModuleList([
        nn.Linear(1024, 1) for _ in range(5)
    ])

    return MyModel(base_model, outputs)

def load_model(model_path):
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = ToTensor()(img)
    img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    return img.unsqueeze(0)  # Adiciona uma dimensão extra para batch

# Carrega o modelo
model = load_model('output/model.pth')

# Carrega e processa a imagem
image_path = 'test_dishes/img_dish_1563811686.jpeg'
image = process_image(image_path)

# Realiza a inferência
with torch.no_grad():
    output = model(image)
    print(output)
