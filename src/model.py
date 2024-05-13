import torch 
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F

class ResnetArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(ResNet50_Weights)
        self.fc1 = nn.Linear(1000, 2)
        self.sigmoid = F.sigmoid
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    

model = ResnetArchitecture()