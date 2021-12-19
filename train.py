from typing import OrderedDict
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import numpy as np
import random
import os
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import models
import datasets
from torch.utils.tensorboard import SummaryWriter

transform = T.Compose([
    T.Resize([75, 75]),
    torchvision.transforms.RandomAffine((-25, 25)),
    T.ToTensor()

])



def get_loss_on_test(model=None):
    status_is_train = model.training
    model.eval()
    n = 1
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    total_loss = 0
    for _ in range(n):

        data = datasets.Small_Dataset(
            root_dir=r'2\test', transform=transform, limit=1)
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


def train(use_l2=False):

    tb = SummaryWriter()
    net = models.Net_18()
    net.pretrain(freeze=True, embad=128)
    net.load_state_dict(torch.load('11_0.model'))
    net.train()
    NEPOCH = 8
    batch_size = 128
    criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    count = 0
    for epoch in range(NEPOCH):
        total_loss = 0

        data = datasets.Big_Dataset(
            r"aligned_images_DB", transform=transform, limit=20)
        data_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size)

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
            tb.add_scalar("Loss", loss.item(), count)
            print(round((count/len1)*100, 2), loss.item())

        lt = get_loss_on_test(net)

        tb.add_scalar("Loss on test", lt, epoch)
        torch.save(net.state_dict(), f'12_{epoch}.model')

    tb.close()


def test():
    # просто для тестирования чего-то
    data = datasets.Big_Dataset(
        root_dir=r'aligned_images_DB', limit=1, transform=transform, max_examples=1000)
    columns = 3
    rows = 1
    for i1 in range(10):
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, columns*rows + 1):

            img = data[i1][i-1].permute(1, 2, 0)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    train(use_l2=1)
    # test()
