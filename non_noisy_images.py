"""
Project:    pytorch_primal_dual
File:       non_noisy_images.py
Created by: louise
On:         01/12/17
At:         12:59 PM
"""
from __future__ import print_function
import os
from os.path import abspath

import numpy as np
import png
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


def create_non_noisy_filelist(img_path):
    abs_path = abspath('.')
    path_img = os.path.join(abs_path, img_path)
    filelist = os.listdir(path_img)
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not (fichier.endswith(".png") or not (fichier.endswith(".jpg"))):
            filelist.remove(fichier)

    return filelist


class NonNoisyImages(Dataset):
    """
    Dataset for noised images and GT.
    """

    def __init__(self, img_path, transform=None):
        self.filelist = create_non_noisy_filelist()
        self.transform = transform

    def __getitem__(self, index):
        print("File : ", self.filelist[index])
        img = Image.open(self.filelist[index])
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        return img

    def __len__(self):
        return len(self.filelist)
