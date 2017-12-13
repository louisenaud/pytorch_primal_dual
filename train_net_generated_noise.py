"""
Project:    pytorch_primal_dual
File:       train_net_generated_noise.py
Created by: louise
On:         01/12/17
At:         12:27 PM
"""

import argparse
import random
import string
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import png
from PIL import Image


import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn


from data_io import NonNoisyImages
from linear_operators import ForwardGradient, ForwardWeightedGradient
from primal_dual_models import Net, GaussianNoiseGenerator


def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def compute_mean_std_data(filelist):
    """

    :param filelist:
    :return:
    """
    tensor_list = []
    for file in filelist:
        img = Image.open(file)
        img_np = np.array(img).ravel()
        tensor_list.append(img_np.ravel())
    pixels = np.concatenate(tensor_list, axis=0)
    return np.mean(pixels), np.std(pixels)

parser = argparse.ArgumentParser(description='Run Primal Dual Net.')
parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Flag to use CUDA, if available')
parser.add_argument('--max_it', type=int, default=20,
                        help='Number of iterations in the Primal Dual algorithm')
parser.add_argument('--max_epochs', type=int, default=20000,
                    help='Number of epochs in the Primal Dual Net')
parser.add_argument('--lambda_rof', type=float, default=3.,
                    help='Step parameter in the ROF model')
parser.add_argument('--theta', type=int, default=0.8,
                    help='Regularization parameter in the Primal Dual algorithm')
parser.add_argument('--tau', type=int, default=0.01,
                    help='Step Parameter in Primal')
parser.add_argument('--save_flag', type=bool, default=True,
                    help='Flag to save or not the result images')
parser.add_argument('--log', type=bool, help="Flag to log loss in tensorboard", default=False)
parser.add_argument('--out_folder', help="output folder for images", default="guillaume_non_norm_20k_epochs/")
args = parser.parse_args()

# Supplemental imports
if args.log:
    from tensorboard import SummaryWriter
    # Keep track of loss in tensorboard
    writer = SummaryWriter()
# Set parameters:
max_epochs = args.max_epochs
max_it = args.max_it
lambda_rof = args.lambda_rof
theta = args.theta
tau = args.tau
#sigma = 1. / (lambda_rof * tau)
sigma = 15.0
batch_size = 1
m, std =122.11/255., 53.55/255.
print(m, std)

# Transform dataset
#transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((m,), (std,))])
transformations = transforms.Compose([transforms.ToTensor()])
dd = NonNoisyImages("/home/louise/src/blog/pytorch_primal_dual/images/BM3D/", transform=transformations)
#m, std = compute_mean_std_dataset(dd.data)
dtype = torch.cuda.FloatTensor
train_loader = DataLoader(dd,
                          batch_size=batch_size,
                          num_workers=1)

m1, n1 = compute_mean_std_data(train_loader.dataset.filelist)
print("m = ", m)
print("s = ", std)
# set up primal and dual variables
img_obs = Variable(train_loader.dataset[0])  # Init img_obs with first image in the data set
x = Variable(img_obs.data.clone().type(dtype))
x_tilde = Variable(img_obs.data.clone().type(dtype))
img_size = img_obs.size()
y = Variable(torch.zeros((img_size[0] + 1, img_size[1], img_size[2])).type(dtype))
y = ForwardGradient().forward(x)
g_ref = y.clone()

# Net approach
w1 = 0.5 * torch.ones([1]).type(dtype)
w2 = 0.4 * torch.ones([1]).type(dtype)
w = Variable(torch.rand(y.size()).type(dtype))
# Primal dual parameters as net parameters
lambda_rof = nn.Parameter(lambda_rof * torch.ones([1]).type(dtype))
sigma = nn.Parameter(sigma * torch.ones([1]).type(dtype))
tau = nn.Parameter(tau * torch.ones([1]).type(dtype))
theta = nn.Parameter(theta*torch.ones([1]).type(dtype))


n_w = torch.norm(w, 2, dim=0)
plt.figure()
plt.imshow(n_w.data.cpu().numpy())
plt.colorbar()
plt.title("Norm of Initial Weights of Gradient of Noised image")

net = Net(w1, w2, w, max_it, lambda_rof, sigma, tau, theta)

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
params = list(net.parameters())
loss_history = []
primal_history = []
dual_history = []
gap_history = []
learning_rate = 1e-2
it = 0
for t in range(max_epochs):
    # Pick random image in dataset
    img_ref = Variable(random.choice(train_loader.dataset)).type(dtype)
    #print(img_ref)
    y = ForwardGradient().forward(img_ref)
    # Pick random noise variance in the given range
    std = np.random.uniform(0.02, 0.2, 1)
    # Apply noise on chosen image
    img_obs = torch.clamp(GaussianNoiseGenerator().forward(img_ref.data, std[0]), min=0.0, max=1.0)
    img_obs = Variable(img_obs).type(dtype)
    x = Variable(img_obs.data.clone())
    w = Variable(torch.rand(y.size()).type(dtype))
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
    if it % 5 == 0 and args.log:
        writer.add_scalar('loss', loss.data[0], it)
    it += 1

    # if args.save_flag:
    #     base_name = id_generator()
    #     folder_name = args.out_folder
    #     if not os.path.isdir(folder_name):
    #         os.mkdir(folder_name)
    #     fn = folder_name + str(it) + "_" + base_name + "_obs.png"
    #     f = open(fn, 'wb')
    #     w1 = png.Writer(img_obs.size()[2], img_obs.size()[1], greyscale=True)
    #     w1.write(f, np.array(transforms.ToPILImage()(img_obs.data.cpu())))
    #     f.close()
    #     fn = folder_name + str(it) + "_" + base_name + "_den.png"
    #     f_res = open(fn, 'wb')
    #     w1.write(f_res, np.array(transforms.ToPILImage()(x_pred.data.cpu())))
    #     f_res.close()

print("w1 = ", net.w1.data[0])
print("w2 = ", net.w2.data[0])
print("tau = ", net.tau.data[0])
print("theta = ", net.theta.data[0])
print("sigma = ", net.sigma.data[0])

std = 0.05
# Apply noise on chosen image
img_obs = Variable(torch.clamp(GaussianNoiseGenerator().forward(img_ref.data, std), min=0., max=1.)).type(dtype)
lin_ref = ForwardWeightedGradient().forward(img_ref.type(dtype), net.w)
grd_ref = ForwardGradient().forward(img_ref)
img_den = net.forward(img_obs, img_obs).type(dtype)
lin_den = ForwardWeightedGradient()(img_den, net.w)
plt.figure()
n1 = torch.norm(lin_ref, 2, dim=0)
plt.imshow(n1.data.cpu().numpy())
plt.title("Linear operator applied on reference image")
plt.figure()
n2 = torch.norm(grd_ref, 2, dim=0)
plt.imshow(n2.data.cpu().numpy())
plt.title("Gradient operator applied on reference image")

n_w = torch.norm(net.w, 2, dim=0)
plt.figure()
plt.imshow(n_w.data.cpu().numpy())
plt.colorbar()
plt.title("Norm of Weights of Gradient of Noised image")

plt.figure()
plt.imshow(np.array(transforms.ToPILImage()((img_obs.data).cpu())))
plt.colorbar()
plt.title("noised image")

plt.figure()
plt.imshow(np.array(transforms.ToPILImage()((x_pred.data).cpu())))
plt.colorbar()
plt.title("denoised image")

# Plot loss
plt.figure()
x = range(len(loss_history))
plt.plot(x, np.asarray(loss_history))
plt.title("Loss")
if args.save_flag:
    plt.savefig('loss.png')


# Validation
# print("Validation-------------------")
# net.eval()
# errors = []
# for t in range(100):
#     # Pick random image in dataset
#     img_ref = Variable(random.choice(train_loader.dataset)).type(dtype)
#     #print(img_ref)
#     y = ForwardGradient().forward(img_ref)
#     # Pick random noise variance in the given range
#     std = np.random.uniform(0.05, 0.2, 1)
#     # Apply noise on chosen image
#     img_obs = torch.clamp(GaussianNoiseGenerator().forward(img_ref.data, std[0]), min=0.0, max=1.0)
#     img_obs = Variable(img_obs).type(dtype)
#     x = Variable(img_obs.data.clone())
#     w = Variable(torch.rand(y.size()).type(dtype))
#     y = ForwardWeightedGradient().forward(x, w)
#
#     # Forward pass: Compute predicted image by passing x to the model
#     x_pred = net(x, img_obs)
#     # Compute and print loss
#     loss = criterion(x_pred, img_ref)
#     errors.append(loss.data[0])
# print(np.mean(errors))

# Test Guillaume data
print("Test-------------------")
net.eval()
errors = []
filelist = []
img_ref = []
img_path = "/media/louise/data/datasets/ForLouLou/"
files = glob.glob(img_path + "*.png")
for fn in files:
    print("Testing on :", fn)
    img_pil = Image.open(fn)
    img_obs = Variable(transforms.ToTensor()(img_pil)).type(dtype)
    y = ForwardGradient().forward(img_obs)

    x = Variable(img_obs.data.clone())
    w = Variable(torch.rand(y.size()).type(dtype))
    y = ForwardWeightedGradient().forward(x, w)

    # Forward pass: Compute predicted image by passing x to the model
    x_pred = net(x, img_obs)

    if args.save_flag:
        base_name = os.path.basename(fn)
        folder_name = args.out_folder
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        w1 = png.Writer(img_obs.size()[2], img_obs.size()[1], greyscale=True)
        fn = folder_name + base_name + "_den.png"
        f_res = open(fn, 'wb')
        w1.write(f_res, np.array(transforms.ToPILImage()(x_pred.data.cpu())))
        f_res.close()
plt.show()
