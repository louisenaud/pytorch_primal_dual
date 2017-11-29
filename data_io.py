"""
Project:    pytorch_primal_dual
File:       data_io.py
Created by: louise
On:         28/11/17
At:         1:22 PM
"""
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class NoisyImages(Dataset):
    """
    Dataset for noised images and GT.
    """
    __xs = []
    __ys = []

    def __init__(self, img_path, transform=transforms.Compose([transforms.ToTensor()])):
        filelist = os.listdir(img_path)
        for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
            if not (fichier.endswith(".png") or not (fichier.endswith(".jpg"))):
                filelist.remove(fichier)

        self.img_path = img_path
        self.filelist = filelist
        self.transform = transform
        self.X = []
        self.Y = []
        self.mean = 0.

        self.mean = torch.mean(self.X)
        self.std = torch.std(self.X)


    def __getitem__(self, index):
        img = Image.open(self.filelist[index])
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1, 1]))
        return img, label



    def __len__(self):
        return len(self.X_train.index)