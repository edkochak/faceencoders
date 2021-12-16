import os
import random
import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class Small_Dataset(Dataset):
    # Для базового, на котором я обучал прошлые

    def __init__(self, root_dir, limit=20, transform=None):
        self.limit = limit
        self.root_dir = root_dir
        self.transform = transform

        self.temp = []
        self.count = 0

        for n, i in enumerate(os.listdir(self.root_dir)):
            if os.path.isdir(self.root_dir+'/'+i) and len(os.listdir(self.root_dir+'/'+i)) >= 2:
                for i1 in os.listdir(self.root_dir+'/'+i):
                    self.temp.append((n, i+'/'+i1))
        random.shuffle(self.temp)
        self.items = self.get_samples()
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def get_samples(self):
        l = []
        for ind1, file1 in self.temp:
            count = 0
            for ind2, file2 in self.temp:
                if ind2 != ind1:
                    continue
                if count > self.limit:
                    break
                for ind3, file3 in self.temp:
                    if ind1 == ind3:
                        continue
                    count += 1
                    if count > self.limit:
                        break
                    l.append([file1, file2, file3])
            self.temp.pop(ind1)
        return l

    def __getitem__(self, idx):
        samples = self.items.pop()
        if self.transform:
            for n, sample in enumerate(samples):
                samples[n] = self.transform(Image.open(
                    os.path.join(self.root_dir, sample)))

        return samples





class Big_Dataset(Dataset):
    # Для огромного датасета
    def __init__(self, root_dir, transform=None, shuffle=False,limit = 20,max_examples = 50000):
        self.root_dir = root_dir
        self.transform = transform
        self.temp = []
        self.limit = limit
        for n, folder in enumerate(os.listdir(self.root_dir)):
            for folder1 in os.listdir(os.path.join(self.root_dir, folder)):
                for file in os.listdir(os.path.join(self.root_dir, folder, folder1)):
                    self.temp.append(
                        (n, os.path.join(self.root_dir, folder, folder1, file)))

        self.temp = self.temp[:max_examples]
        random.shuffle(self.temp)
        self.items = self.get_samples()
        
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def get_samples(self):
        l = []
        for ind1, file1 in self.temp:
            count = 0
            for ind2, file2 in self.temp:
                if ind2 != ind1:
                    continue
                if count > self.limit:
                    break
                for ind3, file3 in self.temp:
                    if ind1 == ind3:
                        continue
                    count += 1
                    if count > self.limit:
                        break
                    l.append([file1, file2, file3])
            self.temp.pop(ind1)
        return l

    def __getitem__(self, idx):
        samples = self.items.pop()
        if self.transform:
            for n, sample in enumerate(samples):
                samples[n] = self.transform(Image.open(sample))

        return samples
