"""
Project:    
File:       primal_dual_model.py
Created by: louise
On:         10/25/17
At:         4:19 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardGradient(nn.Module):
    def __init__(self):
        super(ForwardGradient, self).__init__()

    def forward(self, x, dtype=torch.cuda.FloatTensor):
        """
        
        :param x: PyTorch Variable [1xMxN]
        :param dtype: Tensor type
        :return: PyTorch Variable [2xMxN]
        """
        im_size = x.size()
        gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
        # Horizontal direction
        gradient[0, :, :-1] = x[0, :, 1:] - x[0, :, :-1]
        # Vertical direction
        gradient[1, :-1, :] = x[0, 1:, :] - x[0, :-1, :]
        return gradient


class ForwardWeightedGradient(nn.Module):
    def __init__(self):
        super(ForwardWeightedGradient, self).__init__()

    def forward(self, x, w, dtype=torch.cuda.FloatTensor):
        """
        
        :param x: PyTorch Variable [1xMxN]
        :param w: PyTorch Variable [2xMxN]
        :param dtype: Tensor type
        :return: PyTorch Variable [2xMxN]
        """
        im_size = x.size()
        gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
        # Horizontal direction
        gradient[0, :, :-1] = x[0, :, 1:] - x[0, :, :-1]
        # Vertical direction
        gradient[1, :-1, :] = x[0, 1:, :] - x[0, :-1, :]
        gradient = gradient * w
        return gradient


class BackwardDivergence(nn.Module):
    def __init__(self):
        super(BackwardDivergence, self).__init__()

    def forward(self, y, dtype=torch.cuda.FloatTensor):
        """
        
        :param y: PyTorch Variable, [2xMxN], dual variable
        :param dtype: Tensor type
        :return: PyTorch Variable [1xMxN], divergence
        """
        im_size = y.size()
        # Horizontal direction
        d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_h[0, :, 0] = y[0, :, 0]
        d_h[0, :, 1:-1] = y[0, :, 1:-1] - y[0, :, :-2]
        d_h[0, :, -1] = -y[0, :, -2:-1]

        # Vertical direction
        d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_v[0, 0, :] = y[1, 0, :]
        d_v[0, 1:-1, :] = y[1, 1:-1, :] - y[1, :-2, :]
        d_v[0, -1, :] = -y[1, -2:-1, :]

        # Divergence
        div = d_h + d_v
        return div


class BackwardWeightedDivergence(nn.Module):
    def __init__(self):
        super(BackwardWeightedDivergence, self).__init__()

    def forward(self, y, w, dtype=torch.cuda.FloatTensor):
        """

        :param y: PyTorch Variable, [2xMxN], dual variable
        :param dtype: tensor type
        :return: PyTorch Variable, [1xMxN], divergence
        """
        im_size = y.size()
        y_w = w.cuda() * y
        # Horizontal direction
        d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_h[0, :, 0] = y_w[0, :, 0]
        d_h[0, :, 1:-1] = y_w[0, :, 1:-1] - y_w[0, :, :-2]
        d_h[0, :, -1] = -y_w[0, :, -2:-1]

        # Vertical direction
        d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_v[0, 0, :] = y_w[1, 0, :]
        d_v[0, 1:-1, :] = y_w[1, 1:-1, :] - y_w[1, :-2, :]
        d_v[0, -1, :] = -y_w[1, -2:-1, :]

        # Divergence
        div = d_h + d_v
        return div


class ProximalLinfBall(nn.Module):
    def __init__(self):
        super(ProximalLinfBall, self).__init__()

    def forward(self, p, r, dtype=torch.cuda.FloatTensor):
        """
        
        :param p: PyTorch Variable
        :param r: float
        :param dtype: tensor type
        :return: PyTorch Variable
        """

        m1 = torch.max(torch.add(p.data, - r).type(dtype), torch.zeros(p.size()).type(dtype))
        m2 = torch.max(torch.add(torch.neg(p.data), - r).type(dtype), torch.zeros(p.size()).type(dtype))

        return p - Variable(m1 - m2)


class ProximalL1(nn.Module):
    def __init__(self):
        super(ProximalL1, self).__init__()

    def forward(self, x, f, clambda):
        """
        
        :param x: PyTorch Variable, [1xMxN]
        :param f: PyTorch Variable, [1xMxN]
        :param clambda: float
        :return: PyTorch Variable [1xMxN]
        """
        if x.is_cuda:
            res = x + torch.clamp(f - x, -clambda, clambda).cuda()
        else:
            res = x + torch.clamp(f - x, -clambda, clambda)
        return res


class ProximalL2(nn.Module):
    def __init__(self, x, f, clambda):
        super(ProximalL2, self).__init__()
        self.x = x
        self.f = f
        self.clambda = clambda

    def forward(self):
        return


class PrimalUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalUpdate, self).__init__()
        self.backward_div = BackwardDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs, dtype=torch.cuda.FloatTensor):
        """
        
        :param x: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param img_obs: PyTorch Variable [1xMxN]
        :param dtype: Tensor type
        :return: PyTorch Variable, [1xMxN]
        """
        x = (x + self.tau * self.backward_div.forward(y, dtype) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalWeightedUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalWeightedUpdate, self).__init__()
        self.backward_div = BackwardWeightedDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs, w):
        """

        :param x: PyTorch Variable [1xMxN]
        :param y: PyTorch Variable [2xMxN]
        :param img_obs: PyTorch Variable [1xMxN]
        :return:Pytorch Variable, [1xMxN]
        """
        x = (x + self.tau * self.backward_div.forward(y, w) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalRegularization(nn.Module):
    def __init__(self, theta):
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_tilde, x_old):
        """
        
        :param x: PyTorch Variable, [1xMxN]
        :param x_tilde: PyTorch Variable, [1xMxN]
        :param x_old: PyTorch Variable, [1xMxN]
        :return: PyTorch Variable, [1xMxN]
        """
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde


class DualUpdate(nn.Module):
    def __init__(self, sigma):
        super(DualUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        """
        
        :param x_tilde: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :return: PyTorch Variable, [2xMxN]
        """
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.FloatTensor)
        return y


class DualWeightedUpdate(nn.Module):
    def __init__(self, sigma):
        super(DualWeightedUpdate, self).__init__()
        self.forward_grad = ForwardWeightedGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y, w):
        """
        
        :param x_tilde: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param w: PyTorch Variable, [2xMxN]
        :return: PyTorch Variable, [2xMxN]
        """
        y = y + self.sigma * self.forward_grad.forward(x_tilde, w)
        return y


class PrimalDualNetwork(nn.Module):
    def __init__(self, w, max_it=10, lambda_rof=4.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        """

        :param w: PyTorch Variable, [2xMxN]
        :param max_it: int
        :param lambda_rof: float
        :param sigma: float
        :param tau: float
        :param theta: float
        """
        super(PrimalDualNetwork, self).__init__()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = DualEnergyROF()
        self.pe = 0.0
        self.de = 0.0
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

        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w.cuda())
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w.cuda())
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            # Compute energies
            self.pe = self.energy_primal.forward(x, img_obs.cuda(), self.w, self.clambda)
            self.de = self.energy_dual.forward(y, img_obs, self.w)

        return x_tilde


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


class PrimalEnergyROF(nn.Module):
    def __init__(self):
        """

        """
        super(PrimalEnergyROF, self).__init__()

    def forward(self, x, img_obs, w, clambda):
        """

        :param x: PyTorch Variable, [1xMxN]
        :param img_obs: PyTorch Variable [1xMxN], observed image
        :param w: PyTorch Variable, [2xMxN]
        :param clambda: float, lambda parameter
        :return: PyTorch Variable [1]
        """
        g = ForwardWeightedGradient()
        energy_reg = torch.sum(torch.norm(g.forward(x.cuda(), w), 1))
        energy_data_term = torch.sum(0.5 * clambda * torch.norm(x - img_obs, 2)**2)
        return energy_reg + energy_data_term


class PrimalGeneralEnergyROF(nn.Module):
    def __init__(self):
        super(PrimalGeneralEnergyROF, self).__init__()

    def forward(self, x, b, H, img_obs, clambda):
        """

        :param x: PyTorch Variable, [1xMxN]
        :param img_obs: PyTorch Variable [1xMxN], observed image
        :param clambda: float, lambda parameter
        :return: float, primal ROF energy
        """
        g = ForwardWeightedGradient()
        energy_reg = torch.sum(torch.norm(g.forward(x, torch.cuda.FloatTensor), 1))
        energy_data_term = torch.sum(0.5 * clambda * torch.norm(x - img_obs, 2))
        return energy_reg + energy_data_term


class DualEnergyROF(nn.Module):
    def __init__(self):
        super(DualEnergyROF, self).__init__()

    def forward(self, y, im_obs, w):
        """
        Compute the dual energy of ROF problem.
        :param y: PyTorch Variable, [2xMxN]
        :param im_obs: PyTorch Variable [1xMxN], observed image
        :return: float, dual energy
        """
        d = BackwardWeightedDivergence()
        nrg = -0.5 * (im_obs - d.forward(y, w, torch.cuda.FloatTensor)) ** 2
        nrg = torch.sum(nrg)
        return nrg


class PrimalDualGap(nn.Module):
    def __init__(self):
        super(PrimalDualGap, self).__init__()

    def forward(self, x, y, im_obs, clambda):
        """
        Compute the primal dual gap.
        :param x: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param im_obs: PyTorch Variable, [1xMxN], observed image
        :param clambda: float > 0
        :return: float > 0    w = nn.Parameter(torch.zeros(y.size()))

        """
        dual = DualEnergyROF()
        primal = PrimalEnergyROF()
        g = primal.forward(x, im_obs, clambda) - dual.forward(y, im_obs)
        return g


class MaxMarginLoss(nn.Module):
    def __init__(self):
        super(MaxMarginLoss, self).__init__()

    def forward(self, x, H, b, w, x_gt):
        """
        Custom loss function for the generic ROF problem.
        :param x: PyTorch Variable [1xMxN]
        :param H: PyTorch Variable [1xMxN]
        :param b: PyTorch Variable [1xMxN]
        :param w: PyTorch Variable [1xMxN]
        :return: PyTorch Variable [1xMxN]
        """
        fg = ForwardWeightedGradient()
        output1 = torch.transpose(b, 0, 1) * x + \
                 torch.sum(torch.norm(fg.forward(x, w), 1))
        output2 = max(torch.sum(torch.norm(x_gt - x, 2)) - torch.matmul(torch.transpose(b, 0, 1), x) -
                                                           torch.sum(torch.norm(fg.forward(x, w), 1))
                      , 0.0)
        return output1 + output2


class PrimalGeneralEnergy(nn.Module):
    def __init__(self):
        super(PrimalGeneralEnergy, self).__init__()

    def forward(self, x, H, b, w):
        """
        Primal Energy for the General ROF model.
        :param x: PyTorch Variable [1xMxN]
        :param H: PyTorch Variable [1xMxN]
        :param b: PyTorch Variable [1xMxN]
        :param w: PyTorch Variable [1xMxN]
        :return: PyTorch Variable [1]
        """
        fg = ForwardWeightedGradient()
        nrg1 = torch.sum(0.5 * torch.matmul(torch.transpose(x, 0, 1), torch.matmul(H, x))) + \
               torch.sum(torch.matmul(torch.transpose(b, 0, 1), x))
        nrg2 = torch.sum(torch.norm(fg.forward(x, w), 1))
        return nrg1 + nrg2


class DualGeneralEnergy(nn.Module):
    def __init__(self):
        super(DualGeneralEnergy, self).__init__()

    def forward(self, x, H):
        """
        Dual Energy for the general ROF model.
        :param x: PyTorch Variable [1xMxN]
        :param H: PyTorch Variable [1xM*NxM*N]
        :return: PyTorch Variable, [1]
        """
        nrg = torch.sum(-0.5*torch.matmul(torch.transpose(x), torch.matmul(H, x)))
        return torch.sum(nrg)


class PrimalDualGeneralGap(nn.Module):
    def __init__(self):
        super(PrimalDualGeneralGap, self).__init__()

    def forward(self, x, H, b, w):
        """

        :param x: PyTorch Variable [1xM*N]
        :param H: PyTorch Variable [M*NxMxN]
        :param b: PyTorch Variable [1xM*N]
        :param w: PyTorch Variable [2xM*N]
        :return: Pytorch Variable, [1]
        """
        fg = ForwardWeightedGradient()
        nrg1 = torch.sum(torch.matmul(torch.transpose(b, 0, 1), x))
        nrg2 = torch.sum(torch.norm(fg.forward(x, w), 1))
        nrg3 = torch.sum(torch.matmul(torch.transpose(x), torch.matmul(H, x)))
        return nrg1 + nrg2 + nrg3


class LinearOperator(nn.Module):
    def __init__(self):
        super(LinearOperator, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1).cuda()
        self.conv3 = nn.Conv2d(10, 2, kernel_size=3, stride=1, padding=1).cuda()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        z = Variable(x.data.unsqueeze(0)).cuda()
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        y = Variable(z.data.squeeze(0).cuda())
        return y


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

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


