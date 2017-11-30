"""
Project:    pytorch_primal_dual
File:       primal_dual_models.py
Created by: louise
On:         29/11/17
At:         4:00 PM
"""

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, w1, w2, w, max_it=10, lambda_rof=4.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel

        self.m = nn.ReflectionPad2d(10)
        self.linear_op = LinearOperator()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = DualEnergyROF()
        self.pe = 0.0
        self.de = 0.0
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = lambda_rof

    def forward(self, x, img_obs):
        """

        :param x:
        :param img_obs:
        :return:
        """
        x = img_obs.clone().cuda()
        x_tilde = img_obs.clone().cuda()
        img_size = img_obs.size()
        y = Variable(torch.ones((img_size[0] + 1, img_size[1], img_size[2]))).cuda()
        #y = ForwardGradient().forward(x)

        # Forward pass
        y = self.linear_op(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            # Compute energies
            self.pe = self.energy_primal.forward(x, img_obs.cuda(), self.w, self.clambda)
            self.de = self.energy_dual.forward(y, img_obs, self.w)

        return x


class PrimalDualNetwork_2(nn.Module):
    def __init__(self, w1, w2, w, max_it=10, lambda_rof=4.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        """

        :param w1: PyTorch Variable, [1]
        :param w2: PyTorch Variable, [1]
        :param max_it: int
        :param lambda_rof: float
        :param sigma: float
        :param tau: float
        :param theta: float
        """
        super(PrimalDualNetwork_2, self).__init__()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = DualEnergyROF()
        self.pe = 0.0
        self.de = 0.0
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = lambda_rof

    def forward(self, img_obs):
        """
        Forward function for the PrimalDualNet.
        :param img_obs: PyTorch Variable, [1xMxN]
        :return: PyTorch Variable, [1xMxN]
        """
        x = img_obs.clone().cuda()
        x_tilde = img_obs.clone().cuda()
        img_size = img_obs.size()
        y = Variable(torch.ones((img_size[0] + 1, img_size[1], img_size[2]))).cuda()
        y = ForwardGradient().forward(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            # Compute energies
            self.pe = self.energy_primal.forward(x, img_obs.cuda(), self.w, self.clambda)
            self.de = self.energy_dual.forward(y, img_obs, self.w)

        return x_tilde