"""
Project:    pytorch_primal_dual
File:       create_noisy_images.py
Created by: louise
On:         28/11/17
At:         3:59 PM
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


def noise_gaussian_image(img, sigma_n, dtype=torch.cuda.FloatTensor):
    """

    :param img: PIL image
    :param sigma_n: float, variance of gaussian noise
    :param dtype: type of tensor
    :return:
    """
    # Convert
    img_ref = Variable(transforms.ToTensor()(img))
    noise_v = torch.ones(img_ref.size())
    noise_v = Variable(noise_v.normal_(0.0, sigma_n))
    img_noised = img_ref + noise_v
    return img_noised


def create_noisy_images_dataset(gt_folder, sigma_ns, save=True, folder_images_n="images/noised", dtype=torch.cuda.FloatTensor):
    """

    :param gt_path: absolute path of the GT images folder
    :param sigma_ns: list of variances for gaussian noises
    :param save: bool, flag to save images to disk
    :param folder_images_n: str, folder in which to save the noised images
    :return:
    """
    abs_path = abspath('.')
    path_gt = os.path.join(abs_path, gt_folder)
    filelist = os.listdir(path_gt)
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not (fichier.endswith(".png") or not (fichier.endswith(".jpg"))):
            filelist.remove(fichier)

    # Go through image files
    for fichier in filelist:
        img = Image.open(os.path.join(gt_folder, fichier))
        h, w = img.size
        w1 = png.Writer(h, w, greyscale=True)  # open writer for noised images
        for sigma_n in sigma_ns:
            img_noised = noise_gaussian_image(img, sigma_n, save)
            if save:
                # Create filename for noised images
                base = os.path.splitext(os.path.basename(fichier))[0]
                fn_folder = os.path.join(abs_path, folder_images_n, base)
                if not os.path.exists(fn_folder):
                    os.mkdir(fn_folder)
                fn = os.path.join(fn_folder, base + "_sig_" + str(sigma_n) + ".png")
                print(fn)
                # Open file and save it through png writer
                f = open(fn, 'wb')
                w1.write(f, np.array(transforms.ToPILImage()(img_noised.data.cpu())))
                # Close file
                f.close()

