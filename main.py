"""
Project:    
File:       main.py
Created by: louise
On:         10/9/17
At:         2:25 PM
"""
from __future__ import print_function
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.transforms as transforms


from differential_operators import backward_divergence, forward_gradient
from primal_dual_model import PrimalDualNetwork


def penalization(x):
    return torch.max(x, 0.)


def margin(x1, x2):
    return torch.norm(x1 - x2, 2)


def dual_energy_tvl1(y, im_obs):
    """
    Compute the dual energy of TV-L1 problem.
    :param y: pytorch Variable, [MxNx2]
    :param im_obs: pytorch Variable, observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y, torch.cuda.FloatTensor))**2
    nrg = torch.sum(nrg)
    return nrg


def primal_energy_tvl1(x, img_obs, clambda):
    """

    :param x: pytorch Variables, [MxN]
    :param img_obs: pytorch Variables [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = torch.sum(torch.norm(forward_gradient(x,torch.cuda.FloatTensor), 1))
    energy_data_term = torch.sum(clambda * torch.abs(x - img_obs))
    return energy_reg + energy_data_term


# cuda
use_cuda = torch.cuda.is_available()
print("Cuda = ", use_cuda)
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# images
imsize = 200  # desired size of the output image

loader = transforms.Compose([transforms.Scale(imsize),  # scale imported image
                             transforms.ToTensor()])  # transform it into a torch tensor

if __name__ == '__main__':
    plt.gray()
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()


    t0 = time.time()
    # Create image to noise and denoise
    sigma_n = 0.1
    img_ = Image.open("images/image_Lena512.png")
    h, w = img_.size
    img_ref = Variable(pil2tensor(img_).cuda())
    noise = torch.ones(img_ref.size())
    noise = Variable(noise.normal_(0.0, sigma_n)).cuda()
    img_obs = img_ref + noise

    loader = transforms.Compose([
        transforms.Scale(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    # Parameters
    norm_l = 7.0
    max_it = 200
    theta = 1.0
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    #lambda_TVL1 = 1.0
    lambda_rof = 7.0

    x = Variable(img_obs.data.clone()).cuda()
    x_tilde = Variable(img_obs.data.clone()).cuda()
    img_size = img_ref.size()
    y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2]))).cuda()

    # Net approach
    w = nn.Parameter(torch.zeros(y.size()))
    net = PrimalDualNetwork(w)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    loss_history = []
    for t in range(800):
        # Forward pass: Compute predicted image by passing x to the model
        x_pred = net(x)
        # Compute and print loss
        loss = criterion(x_pred, img_ref)
        print(t, loss.data[0])
        loss_history.append(loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    ax1.imshow(img_)
    ax1.set_title("Reference image")
    ax2.imshow(tensor2pil(img_obs.data.cpu()))
    ax2.set_title("Observed image")
    ax3.imshow(tensor2pil(x_pred.data.cpu()))
    ax3.set_title("Denoised image")

    # Test image
    img_t = Image.open("images/image_Barbara512.png")
    h, w = img_t.size
    img_ref_t = Variable(pil2tensor(img_t).cuda())
    noise_2 = torch.ones(img_ref_t.size())
    noise_2 = Variable(noise_2.normal_(0.0, sigma_n)).cuda()
    img_obs_t = img_ref_t + noise
    img_dn = net.forward(img_obs_t)

    f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax21.imshow(img_t)
    ax21.set_title("Reference image")
    ax22.imshow(img_obs_t.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax22.set_title("Observed image")
    ax23.imshow(img_dn.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax23.set_title("Denoised image")

    # Plot loss
    plt.figure()
    x = range(len(loss_history))
    plt.plot(x, np.asarray(loss_history))
    plt.show()
