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
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class NoisyImages(Dataset):
    """
    Dataset for noised images and GT.
    """

    def __init__(self, img_path, transform=None):
        filelist = []
        img_ref = []
        for root, dirs, files in os.walk(img_path):
            path = root.split(os.sep)
            print(os.path.basename(root))
            for dir in dirs:
                print(dir)
                filess = os.listdir(os.path.join(root, dir))
                for file in filess:
                    print(file)
                    if (file.endswith(".png") or (file.endswith(".jpg"))):
                        filelist.append(os.path.join(root, dir, file))
                        img_ref_path = os.path.join(root + "../GT/", dir + ".png")
                        img_ref.append(img_ref_path)
        self.transform = transform
        self.X = filelist
        self.Y = img_ref
        self.data_list=[]
        result = [y for x in os.walk("./images") for y in glob(os.path.join(x[0], '*.png'))]
        print(result)
        data_ims = torch.zeros(len(result), 1, 512, 512)
        data_ref = torch.zeros(len(result), 1, 512, 512)
        for k, fn in enumerate(self.X):
            img = Image.open(fn)
            data_ims[k, :, :, :] = transforms.ToTensor()(img)
            im_ref = Image.open(self.Y[k])
            data_ref[k, :, :, :] = transforms.ToTensor()(im_ref)
            self.data_list.append((transforms.ToTensor()(img),  transforms.ToTensor()(im_ref)))
        self.size = transforms.ToTensor()(img).size()
        self.data = data_ims
        self.data_ref = data_ref

    def __getitem__(self, index):
        print("File : ", self.X[index])
        img = Image.open(self.X[index])
        img_ref = Image.open(self.Y[index])
        if self.transform is not None:
            img = self.transform(img)
            img_ref = self.transform(img_ref)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(img_ref))
        return img, label

    def __len__(self):
        return len(self.X)


def compute_mean_std_dataset(data):
    """
    Function to compute the mean and stds of a Pytorch dataset.
    :param dataset: pytorch dataset of images.
    :return: list of means and stds for each channel of the input images.
    """
    means = []
    stdevs = []

    for i in range(data.shape[1]):
        pixels = data[:, i, :, :].numpy().ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    return means, stdevs


def create_non_noisy_filelist(img_path):
    """

    :param img_path:
    :return:
    """
    abs_path = os.path.abspath('.')
    path_img = os.path.join(abs_path, img_path)
    print(path_img)
    filelist = os.listdir(path_img)
    img_files = []
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if fichier.endswith(".png") or (fichier.endswith(".jpg")):
            img_files.append(os.path.join(path_img, fichier))

    return img_files


class NonNoisyImages(Dataset):
    """
    Dataset for reference images to be noised on the fly in the network.
    """

    def __init__(self, img_path, transform=None):
        self.filelist = create_non_noisy_filelist(img_path)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.filelist[index])
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.filelist)
