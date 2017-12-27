"""
Project:    pytorch_primal_dual
File:       primal_dual_models.py
Created by: louise
On:         29/11/17
At:         4:00 PM
"""
import numpy as np
from numpy import random
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from primal_dual_updates import PrimalWeightedUpdate, PrimalRegularization, DualWeightedUpdate
from proximal_operators import ProximalLinfBall
from linear_operators import GeneralLinearOperator, GeneralLinearAdjointOperator


class LinearOperator(nn.Module):
    def __init__(self):
        """
        Constructor of the learnable weight parameter CNN.
        """
        super(LinearOperator, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1).cuda()
        self.conv3 = nn.Conv2d(10, 2, kernel_size=3, stride=1, padding=1).cuda()

    def forward(self, x):
        """
        Function to learn the Linear Operator L with a small CNN.
        :param x: PyTorch Variable [1xMxN], primal variable.
        :return: PyTorch Variable [2xMxN], output of learned linear operator
        """
        z = Variable(x.data.unsqueeze(0)).cuda()
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        y = Variable(z.data.squeeze(0).cuda())
        return y


class GaussianNoiseGenerator(nn.Module):
    def __init__(self):

        super(GaussianNoiseGenerator, self).__init__()

    def forward(self, img, std, mean=0.0, dtype=torch.cuda.FloatTensor):
        """
        Function to add gaussian noise with zero mean and given std to img.
        :param img: PyTorch Variable [1xMxN], image to noise.
        :param std: PyTorch tensor [1]
        :param mean: float
        :param dtype: Pytorch Tensor type, def=torch.cuda.FloatTensor
        :return: Pytorch variable [1xMxN], noised img.
        """
        noise = torch.zeros(img.size()).type(dtype)
        noise.normal_(mean, std=std)
        img_n = img + noise
        return img_n


class PoissonNoiseGenerator(nn.Module):
    def __init__(self):
        super(PoissonNoiseGenerator, self).__init__()

    def forward(self, img, param=500., dtype=torch.cuda.FloatTensor):
        """
        Function to create random Poisson noise on an image.
        :param img:
        :param param:
        :param dtype:
        :return:
        """
        img_np = np.array(transforms.ToPILImage()(img.data.cpu()))
        poissonNoise = random.poisson(param, img_np.shape).astype(float)

        noisy_img = img + poissonNoise
        noisy_img_pytorch = Variable(transforms.ToTensor()(noisy_img).type(dtype))
        return noisy_img_pytorch


class Net(nn.Module):

    def __init__(self, w1, w2, w, max_it, lambda_rof, sigma, tau, theta, dtype=torch.cuda.FloatTensor):
        """
        Constructor of the Primal Dual Net.
        :param w1: Pytorch variable [2xMxN]
        :param w2: Pytorch variable [2xMxN]
        :param w: Pytorch variable [2xMxN]
        :param max_it: int
        :param lambda_rof: float
        :param sigma: float
        :param tau: float
        :param theta: float
        :param dtype: Pytorch Tensor type, torch.cuda.FloatTensor by default.
        """
        super(Net, self).__init__()
        self.linear_op = LinearOperator()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.pe = 0.0
        self.de = 0.0
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = nn.Parameter(lambda_rof.data)
        self.sigma = nn.Parameter(sigma.data)
        self.tau = nn.Parameter(tau.data)
        self.theta = nn.Parameter(theta.data)

        self.type = dtype

    def forward(self, x, img_obs):
        """
        Forward function for the Net model.
        :param x: Pytorch variable [1xMxN]
        :param img_obs: Pytorch variable [1xMxN]
        :return: Pytorch variable [1xMxN]
        """
        x = Variable(img_obs.data.clone()).cuda()
        x_tilde = Variable(img_obs.data.clone()).cuda()
        img_size = img_obs.size()
        y = Variable(torch.ones((img_size[0] + 1, img_size[1], img_size[2]))).cuda()

        # Forward pass
        y = self.linear_op(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        self.w.type(self.type)
        self.theta.data.clamp_(0, 5)
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y.data.clamp_(0, 1)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            x.data.clamp_(0, 1)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            x_tilde.data.clamp_(0, 1)

        return x


class GeneralNet(nn.Module):

    def __init__(self, w1, w2, w, max_it, lambda_rof, sigma, tau, theta, dtype=torch.cuda.FloatTensor):
        """
        Constructor of the Primal Dual Net.
        :param w1: Pytorch variable [2xMxN]
        :param w2: Pytorch variable [2xMxN]
        :param w: Pytorch variable [2xMxN]
        :param max_it: int
        :param lambda_rof: float
        :param sigma: float
        :param tau: float
        :param theta: float
        :param dtype: Pytorch Tensor type, torch.cuda.FloatTensor by default.
        """
        super(Net, self).__init__()
        self.linear_op = GeneralLinearOperator()
        self.linear_op_adj = GeneralLinearAdjointOperator()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.pe = 0.0
        self.de = 0.0
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = nn.Parameter(lambda_rof.data)
        self.sigma = nn.Parameter(sigma.data)
        self.tau = nn.Parameter(tau.data)
        self.theta = nn.Parameter(theta.data)

        self.type = dtype

    def forward(self, x, img_obs):
        """
        Forward function for the Net model.
        :param x: Pytorch variable [1xMxN]
        :param img_obs: Pytorch variable [1xMxN]
        :return: Pytorch variable [1xMxN]
        """
        x = Variable(img_obs.data.clone()).cuda()
        x_tilde = Variable(img_obs.data.clone()).cuda()
        img_size = img_obs.size()
        y = Variable(torch.ones((img_size[0] + 1, img_size[1], img_size[2]))).cuda()

        # Forward pass
        y = self.linear_op(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        self.w.type(self.type)
        self.theta.data.clamp_(0, 5)
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y.data.clamp_(0, 1)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            x.data.clamp_(0, 1)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            x_tilde.data.clamp_(0, 1)

        return x

