from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision

import random
import os
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import datasets
from torch.utils.tensorboard import SummaryWriter

transform = T.Compose([
    T.CenterCrop((170,170)),
    T.Resize([100, 100]),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
])



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=True, embad=50, freeze=False):
        # self.load_state_dict(torch.load('models/9.model'))
        if new_fc:
            self.model.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(2048,3000),nn.ReLU(),nn.Linear(3000, embad))
        if freeze:
            freeze_names = ['layer4', 'avgpool', 'fc']

            for name, child in self.model.named_children():
                if name not in freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False


def get_loss_on_test(model=None):
    status_is_train = model.training
    model.eval()
    n = 1
    criterion = nn.TripletMarginLoss(margin=10, p=2)
    total_loss = 0
    for _ in range(n):

        data =datasets.Small_Dataset(
            root_dir=r'2\test', transform=transform,limit=1)
        data_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=256)
        count = 0

        for i in data_loader:
            count += 1
            out = torch.nn.functional.normalize(model(i[0]))
            out1 = torch.nn.functional.normalize(model(i[1]))
            out2 = torch.nn.functional.normalize(model(i[2]))
            loss = criterion(out, out1, out2)
            total_loss += loss.item()

    if status_is_train:
        model.train()
    return total_loss


def train(use_l2 = False):

    tb = SummaryWriter()
    net = Net()
    net.pretrain(freeze=True,embad=50)
    net.load_state_dict(torch.load('8_0.model'))
    net.train()
    NEPOCH = 8
    criterion = nn.TripletMarginLoss(margin=10, p=2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    count = 0
    for epoch in range(NEPOCH):
        total_loss = 0

        data =datasets.Small_Dataset(r"2\train",transform = transform, limit=20)
        data_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=128)

        len1 = len(data_loader)

        for i in data_loader:
            count += 1
            optimizer.zero_grad()
            out = torch.nn.functional.normalize(net(i[0]))
            out1 = torch.nn.functional.normalize(net(i[1]))
            out2 = torch.nn.functional.normalize(net(i[2]))
            
            loss = criterion(out, out1, out2)
            if use_l2:
                l2_lambda = 0.01
                l2_reg = torch.tensor(0.)
                for param in net.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            print(round((count/len1)*100, 2), loss.item())
            
        lt= get_loss_on_test(net)
        tb.add_scalar("Loss", loss.item(), count)
        tb.add_scalar("Loss on test",lt , count)
        torch.save(net.state_dict(), f'9_{epoch}.model')
        
        
    tb.close()
    


def test():
    # просто для тестирования чего-то
    data = datasets.Small_Dataset(root_dir=r'2\test', transform=transform,limit=1)
    data_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=50, shuffle=True)
    images = data[0]

    for i in range(10):
        plt.imshow(  data[i][0].permute(1, 2, 0)  )
        plt.show()
        

    


if __name__ == '__main__':
    train()
    # test()
