"""
Project:    
File:       main.py
Created by: louise
On:         10/9/17
At:         2:25 PM
"""
from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

import png
import argparse
import time

from primal_dual_model import PrimalDualNetwork, ForwardGradient, BackwardDivergence, ForwardWeightedGradient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Primal Dual Net.')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Flag to use CUDA, if available')
    parser.add_argument('--max_it', type=int, default=15,
                        help='Number of iterations in the Primal Dual algorithm')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Number of epochs in the Primal Dual Net')
    parser.add_argument('--lambda_rof', type=float, default=5.,
                        help='Lambda parameter in the ROF model')
    parser.add_argument('--theta', type=int, default=0.8,
                        help='Regularization parameter in the Primal Dual algorithm')
    parser.add_argument('--tau', type=int, default=0.01,
                        help='Parameter in Primal')
    parser.add_argument('--save_flag', type=bool, default=True,
                        help='Flag to save or not the result images')
    parser.add_argument('--sigma_n', type=float, default=0.01,
                        help='Noise variance to add on reference image.')

    args = parser.parse_args()

    max_epochs = args.max_epochs
    max_it = args.max_it
    lambda_rof = args.lambda_rof
    theta = args.theta
    tau = args.tau
    sigma = 1. / (lambda_rof * tau)
    # cuda
    use_cuda = (torch.cuda.is_available() and args.use_cuda)
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Transforms PIL <-> PyTorch tensors
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Create image to noise and denoise / train the network on
    sigma_n = args.sigma_n  # Noise variance
    img_ = Image.open("images/image_Lena512.png")
    h, w = img_.size
    img_ref = Variable(pil2tensor(img_).type(dtype))
    noise = torch.ones(img_ref.size())
    noise = Variable(noise.normal_(0.0, sigma_n).type(dtype))
    img_obs = img_ref + noise
    img_n = img_obs.data.cpu().numpy().reshape((512, 512))

    # Parameters
    norm_l = 7.0
    max_it = 200
    theta = 0.8
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    lambda_rof = 8.0  # 7.0

    x = Variable(img_obs.data.clone().type(dtype))
    x_tilde = Variable(img_obs.data.clone().type(dtype))
    img_size = img_ref.size()
    y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2])).type(dtype))
    y = ForwardGradient().forward(x)
    g_ref = y.clone()

    # Net approach
    w = nn.Parameter(torch.rand(y.size()).type(dtype))
    n_w = torch.norm(w, 1, dim=0)
    plt.figure()
    plt.imshow(n_w.data.cpu().numpy())
    plt.colorbar()

    net = PrimalDualNetwork(w, max_it=max_it, lambda_rof=lambda_rof, sigma=sigma, tau=tau, theta=theta)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    params = list(net.parameters())
    loss_history = []
    primal_history = []
    dual_history = []
    gap_history = []
    learning_rate = 1e-4
    for t in range(max_epochs):
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
        # Compute energies and store them for plotting
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

    # Save results if specified so:
    if args.save_flag:
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
    img_ref_t = Variable(pil2tensor(img_t).type(dtype))
    noise_2 = torch.ones(img_ref_t.size())
    noise_2 = Variable(noise_2.normal_(0.0, sigma_n).type(dtype))
    img_obs_t = img_ref_t + noise
    t_0 = time.time()
    img_dn = net.forward(img_obs_t)
    t_1 = time.time()
    print("Elapsed time: ", t_1 - t_0, "s")

    f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax21.imshow(np.array(img_t))
    ax21.set_title("Reference image")
    ax22.imshow(img_obs_t.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax22.set_title("Observed image")
    ax23.imshow(img_dn.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax23.set_title("Denoised image")
    # Save results if specified so:
    if args.save_flag:
        f = open('barbara_noised.png', 'wb')
        w1 = png.Writer(512, 512, greyscale=True)
        w1.write(f, np.array(tensor2pil(img_obs_t.data.cpu())))
        f.close()
        f_res = open('barbara_denoised.png', 'wb')
        w1.write(f_res, np.array(tensor2pil(img_dn.data.cpu())).reshape(512, 512))
        f_res.close()

    # plt.figure()
    # res = ForwardWeightedGradient().forward(x, w)
    # plt.imshow(res.data[0].cpu().numpy())
    # plt.title("Gradient wrt x")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(g_ref.data[0].cpu().numpy())
    # plt.title("Gradient WRT x")
    #
    # plt.figure()
    # plt.imshow(w.data[1].cpu().numpy())
    # plt.title("Gradient wrt y")
    # plt.colorbar()


    # Test image
    img_t = Image.open("images/image_Boats512.png")
    h, w1 = img_t.size
    img_ref_t = Variable(pil2tensor(img_t).type(dtype))
    noise_2 = torch.ones(img_ref_t.size())
    noise_2 = Variable(noise_2.normal_(0.0, sigma_n).type(dtype))
    img_obs_t = img_ref_t + noise
    t_0 = time.time()
    img_dn = net.forward(img_obs_t)
    t_1 = time.time()
    print("Elapsed time: ", t_1 - t_0, "s")

    f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax21.imshow(np.array(img_t))
    ax21.set_title("Reference image")
    ax22.imshow(img_obs_t.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax22.set_title("Observed image")
    ax23.imshow(img_dn.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax23.set_title("Denoised image")
    # Save results if specified so:
    if args.save_flag:
        f = open('boats_noised.png', 'wb')
        w1 = png.Writer(512, 512, greyscale=True)
        w1.write(f, np.array(tensor2pil(img_obs_t.data.cpu())))
        f.close()
        f_res = open('boats_denoised.png', 'wb')
        w1.write(f_res, np.array(tensor2pil(img_dn.data.cpu())).reshape(512, 512))
        f_res.close()

    # Test image
    img_t = Image.open("images/image_Lake512.png")
    h, w1 = img_t.size
    img_ref_t = Variable(pil2tensor(img_t).type(dtype))
    noise_2 = torch.ones(img_ref_t.size())
    noise_2 = Variable(noise_2.normal_(0.0, sigma_n).type(dtype))
    img_obs_t = img_ref_t + noise
    t_0 = time.time()
    img_dn = net.forward(img_obs_t)
    t_1 = time.time()
    print("Elapsed time: ", t_1 - t_0, "s")

    f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax21.imshow(np.array(img_t))
    ax21.set_title("Reference image")
    ax22.imshow(img_obs_t.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax22.set_title("Observed image")
    ax23.imshow(img_dn.data.cpu().mul_(255).numpy().reshape(512, 512))
    ax23.set_title("Denoised image")
    # Save results if specified so:
    if args.save_flag:
        f = open('lake_noised.png', 'wb')
        w1 = png.Writer(512, 512, greyscale=True)
        w1.write(f, np.array(tensor2pil(img_obs_t.data.cpu())))
        f.close()
        f_res = open('lake_denoised.png', 'wb')
        w1.write(f_res, np.array(tensor2pil(img_dn.data.cpu())).reshape(512, 512))
        f_res.close()

    # Plot loss
    plt.figure()
    x = range(len(loss_history))
    plt.plot(x, np.asarray(loss_history))
    plt.title("Loss")

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
