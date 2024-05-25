### This file initializes the model we will be using to complete the task

import torch 
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F

class ResnetArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        self.fc1 = nn.Linear(1000, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

model = ResnetArchitecture()