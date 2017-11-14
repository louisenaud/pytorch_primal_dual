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

import png
import prox_tv

from primal_dual_model import PrimalDualNetwork, ForwardGradient, BackwardDivergence


def penalization(x):
    return torch.max(x, 0.)


def margin(x1, x2):
    return torch.norm(x1 - x2, 2)


def test_operators_adjoints(x, y):
    fg = ForwardGradient()
    bd = BackwardDivergence()
    gradx = fg.forward(x).cuda()
    divy = bd.forward(y).cuda()
    return torch.sum(gradx * y + divy * x)


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

    # Create image to noise and denoise
    sigma_n = 0.1
    img_ = Image.open("images/image_Lena512.png")
    h, w = img_.size
    img_ref = Variable(pil2tensor(img_).cuda())
    noise = torch.ones(img_ref.size())
    noise = Variable(noise.normal_(0.0, sigma_n)).cuda()
    img_obs = img_ref + noise
    img_n = img_obs.data.cpu().numpy().reshape((512, 512))

    loader = transforms.Compose([
        transforms.Scale(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    # Parameters
    norm_l = 7.0
    max_it = 200
    theta = 0.8
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    #lambda_TVL1 = 1.0
    lambda_rof = 8.0  # 7.0

    x = Variable(img_obs.data.clone()).cuda()
    x_tilde = Variable(img_obs.data.clone()).cuda()
    img_size = img_ref.size()
    y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2]))).cuda()
    y = ForwardGradient().forward(x)

    # Net approach
    w = nn.Parameter(torch.randn(y.size()).cuda())
    n_w = torch.norm(w, 1, dim=0)
    plt.figure()
    plt.imshow(n_w.data.cpu().numpy())
    plt.colorbar()

    net = PrimalDualNetwork(w)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    loss_history = []
    primal_history = []
    dual_history = []
    gap_history = []
    learning_rate = 1e-4
    for t in range(200):
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
        # Compute energies
        pe = net.pe.data[0]
        de = net.de.data[0]
        primal_history.append(pe)
        dual_history.append(de)
        gap_history.append(pe - de)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    ax1.imshow(np.array(img_))
    ax1.set_title("Reference image")
    ax2.imshow(np.array(tensor2pil(img_obs.data.cpu())).reshape(512, 512))
    ax2.set_title("Observed image")
    ax3.imshow(np.array(tensor2pil(x_pred.data.cpu())).reshape(512, 512))
    ax3.set_title("Denoised image")
    ax4.imshow(np.abs(np.array(tensor2pil(img_obs.data.cpu())) - np.array(tensor2pil(x_pred.data.cpu()))))
    print(gap_history[-1])

    f = open('lena_noised.png', 'wb')
    w1 = png.Writer(512, 512, greyscale=True)
    w1.write(f, np.array(tensor2pil(img_obs.data.cpu())))
    f.close()
    f_res = open('lena_denoised.png', 'wb')
    w1.write(f_res, np.array(tensor2pil(x_pred.data.cpu())).reshape(512, 512))
    f_res.close()
    # Test image
    img_t = Image.open("images/image_Barbara512.png")
    h, w1 = img_t.size
    img_ref_t = Variable(pil2tensor(img_t).cuda())
    noise_2 = torch.ones(img_ref_t.size())
    noise_2 = Variable(noise_2.normal_(0.0, sigma_n)).cuda()
    img_obs_t = img_ref_t + noise
    img_dn = net.forward(img_obs_t)

    f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax21.imshow(np.array(img_t))
    ax21.set_title("Reference image")
    ax22.imshow(img_obs_t.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax22.set_title("Observed image")
    ax23.imshow(img_dn.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax23.set_title("Denoised image")

    #n_w = torch.norm(w, 1, dim=0)
    np_w = w.data.cpu().numpy()
    #latestDs = np_w / np.sum(np_w, axis=1)[:, None]
    row_sums = np_w.sum(axis=0)
    latestDs = np_w / row_sums[:, :, np.newaxis]
    I = np.argsort(latestDs.dot(np.arange(latestDs.shape[1])))
    latestDs = np_w[I]
    plt.figure()
    plt.imshow(latestDs)
    plt.colorbar()

    # Plot loss
    plt.figure()
    x = range(len(loss_history))
    plt.plot(x, np.asarray(loss_history))

    plt.figure()
    x = range(len(primal_history))
    plt.plot(x, np.asarray(primal_history))
    plt.title("Primal")

    plt.figure()
    x = range(len(dual_history))
    plt.plot(x, np.asarray(dual_history))
    plt.title("Dual")

    plt.figure()
    x = range(len(gap_history))
    plt.plot(x, np.asarray(gap_history))
    plt.title("Gap")
    plt.show()
