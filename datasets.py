import os
import random
import numpy
import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import mediapipe as mp
import cv2 as cv
import numpy as np


class Small_Dataset(Dataset):
    # Для базового, на котором я обучал прошлые

    def __init__(self, root_dir, limit=20, transform=None):
        self.limit = limit
        self.root_dir = root_dir
        self.transform = transform
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1)
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

    def remove_bg(self, frame):
        frame = np.array(frame)
        height, width, channel = frame.shape
        RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(RGB)
        mask = results.segmentation_mask
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.6
        output_image = np.where(condition, frame, frame*0)
        img = Image.fromarray(output_image.copy())
        return img

    def get_samples(self):
        l = []
        for n, (ind1, file1) in enumerate(self.temp):
            count = 0
            for ind2, file2 in self.temp[n:]:
                if ind2 != ind1 or file1 == file2:
                    continue
                if count > self.limit:
                    break
                for ind3, file3 in self.temp[n:]:
                    if ind1 == ind3:
                        continue
                    count += 1
                    if count > self.limit:
                        break
                    l.append([file1, file2, file3])
            self.temp.pop(ind1)
        return l

    def __getitem__(self, idx):
        samples = self.items[idx].copy()
        if self.transform:
            for n, sample in enumerate(samples):
                samples[n] = self.transform(self.remove_bg(Image.open(
                    os.path.join(self.root_dir, sample))))

        return samples


class Big_Dataset(Dataset):
    # Для огромного датасета
    def __init__(self, root_dir, transform=None, shuffle=False, limit=20, max_examples=50000):
        self.root_dir = root_dir
        self.transform = transform
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1)
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

    def remove_bg(self, frame):
        frame = np.array(frame)
        height, width, channel = frame.shape
        RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(RGB)
        mask = results.segmentation_mask
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.8
        output_image = np.where(condition, frame, frame*0)
        img = Image.fromarray(output_image.copy())
        return img

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
        samples = self.items[idx].copy()
        if self.transform:
            for n, sample in enumerate(samples):
                samples[n] = self.transform(self.remove_bg(Image.open(sample)))

        return samples
