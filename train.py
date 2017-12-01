"""
Project:    pytorch_primal_dual
File:       train.py
Created by: louise
On:         30/11/17
At:         1:42 PM
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboard import SummaryWriter

from data_io import NoisyImages, compute_mean_std_dataset
from linear_operators import ForwardGradient, ForwardWeightedGradient, BackwardWeightedDivergence
from primal_dual_models import Net


parser = argparse.ArgumentParser(description='Run Primal Dual Net.')
parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Flag to use CUDA, if available')
parser.add_argument('--max_it', type=int, default=20,
                        help='Number of iterations in the Primal Dual algorithm')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='Number of epochs in the Primal Dual Net')
parser.add_argument('--lambda_rof', type=float, default=7.,
                    help='Lambda parameter in the ROF model')
parser.add_argument('--theta', type=int, default=0.9,
                    help='Regularization parameter in the Primal Dual algorithm')
parser.add_argument('--tau', type=int, default=0.01,
                    help='Parameter in Primal')
parser.add_argument('--save_flag', type=bool, default=True,
                    help='Flag to save or not the result images')
args = parser.parse_args()

# Set parameters:
max_epochs = args.max_epochs
max_it = args.max_it
lambda_rof = args.lambda_rof
theta = args.theta
tau = args.tau
sigma = 1. / (lambda_rof * tau)
batch_size = 1
m, std = 0.47, 0.21
# Keep track of loss in tensorboard
writer = SummaryWriter()
# Transform dataset
transformations = transforms.Compose([transforms.Normalize(m, std), transforms.ToTensor()])
dd = NoisyImages("/home/louise/src/blog/pytorch_primal_dual/images/noised/", transform=transformations)
#m, std = compute_mean_std_dataset(dd.data)
dtype = torch.cuda.FloatTensor
train_loader = DataLoader(dd,
                          batch_size=batch_size,
                          num_workers=4
                          )
for img_obs, img_ref in train_loader.dataset.data_list:
    print(img_obs.size())
    if args.use_cuda:
        img_obs, img_ref = img_obs.cuda(), img_ref.cuda()
    img_obs, img_ref = Variable(img_obs), Variable(img_ref)


rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
# Go through each split
for train_index, test_index in rs.split(train_loader.dataset.data_list):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train = np.array(train_loader.dataset.data_list)[train_index][0]
    X_test = np.array(train_loader.dataset.data_list)[test_index][0]
    y_train = np.array(train_loader.dataset.data_list)[train_index][1]
    y_test = np.array(train_loader.dataset.data_list)[test_index][1]
    # set up primal and dual variables
    img_obs = Variable(train_loader.dataset.data_list[0][0])
    x = Variable(img_obs.data.clone().type(dtype))
    x_tilde = Variable(img_obs.data.clone().type(dtype))
    img_size = img_obs.size()
    y = Variable(torch.zeros((img_size[0] + 1, img_size[1], img_size[2])).type(dtype))
    y = ForwardGradient().forward(x)
    g_ref = y.clone()

    # Net approach
    w1 = 0.5 * torch.ones([1]).type(dtype)
    w2 = 0.5 * torch.ones([1]).type(dtype)
    w = Variable(torch.rand(y.size()).type(dtype))

    # w = nn.Parameter(torch.rand(y.size()).type(dtype))
    n_w = torch.norm(w, 2, dim=0)
    plt.figure()
    plt.imshow(n_w.data.cpu().numpy())
    plt.colorbar()
    plt.title("Norm of Gradient of Noised image")

    net = Net(w1, w2, w, max_it=max_it, lambda_rof=lambda_rof, sigma=sigma, tau=tau, theta=theta)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    params = list(net.parameters())
    loss_history = []
    primal_history = []
    dual_history = []
    gap_history = []
    learning_rate = 1e-4
    it = 0
    for t in range(max_epochs):
        for img_obs, img_ref in X_train, y_train:
            if args.use_cuda:
                img_obs, img_ref = img_obs.cuda(), img_ref.cuda()
            img_obs, img_ref = Variable(img_obs), Variable(img_ref)

            y = ForwardWeightedGradient().forward(x, w)
            # Forward pass: Compute predicted image by passing x to the model
            x_pred = net(x, img_obs)
            # Compute and print loss
            loss = criterion(x_pred, img_ref)
            loss_history.append(loss.data[0])
            print(t, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % 100 == 0:
                writer.add_scalar('loss', loss.data[0], it)
            it += 1

        # Compute energies and store them for plotting
        # pe = net.pe.data[0]
        # de = net.de.data[0]
        # primal_history.append(pe)
        # dual_history.append(de)
        # gap_history.append(pe - de)

    for img_obs, img_ref in X_test, y_test:
        x = Variable(img_obs).type(dtype)
        img_obs = Variable(img_obs).type(dtype)
        x_pred = net.forward(img_obs, img_obs)
        plt.figure()
        plt.imshow(x_pred.data.cpu().numpy().reshape(x.size()[1], x.size()[2]))
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(train_loader.dataset.data, train_loader.dataset.data_ref, test_size=0.33, random_state=42)

