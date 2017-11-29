"""
Project:    pytorch_primal_dual
File:       create_noisy_images_test.py
Created by: louise
On:         28/11/17
At:         4:37 PM
"""
import unittest
import os
from PIL import Image

import torch
from torch.autograd import Variable

from create_noisy_images import noise_gaussian_image, create_noisy_images_dataset


class TestNoiseMethods(unittest.TestCase):
    def test_noise_image(self):
        path = os.path.join(".", "images", "GT", "image_Boats512.png")
        img = Image.open(path)
        img_n = noise_gaussian_image(img, 0.1)

        self.assertEquals(img.size[0], img_n.size()[1])

    def test_create_dataset(self):
        path = os.path.join("images", "GT")
        path_noise = os.path.join("images", "noised")
        print path_noise
        if not os.path.exists(path_noise):
            os.mkdir(path_noise)

        create_noisy_images_dataset(path, [0.01, 0.05, 0.1], save=True, folder_images_n=path_noise)

        self.assertTrue(os.path.exists(os.path.join(path_noise, "image_Boats512", "image_Boats512_sig_" + str(0.1) + ".png")))

