from typing import OrderedDict
import torch
from torchvision import models
import torch.nn as nn

class Net_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=True, embad=50, freeze=False):
        # self.load_state_dict(torch.load('models/9.model'))
        if new_fc:
            self.model.fc = nn.Sequential(nn.Dropout(p=0.1),nn.Linear(512, embad))
        if freeze:
            freeze_names = ['layer4', 'avgpool', 'fc']

            for name, child in self.model.named_children():
                if name not in freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False




class Net_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=True, embad=50, freeze=False):
        # self.load_state_dict(torch.load('models/9.model'))
        if new_fc:
            self.model.fc = nn.Sequential(OrderedDict([('1', nn.Dropout(p=0.1)), ('2', nn.Linear(
                2048, 3000)), ('3', nn.Dropout(p=0.1)), ('4', nn.ReLU()), ('5', nn.Linear(3000, embad))]))
        if freeze:
            freeze_names = ['layer4', 'avgpool', 'fc']

            for name, child in self.model.named_children():
                if name not in freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False