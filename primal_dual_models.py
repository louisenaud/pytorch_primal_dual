"""
Project:    pytorch_primal_dual
File:       primal_dual_models.py
Created by: louise
On:         29/11/17
At:         4:00 PM
"""
import time

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


from primal_dual_updates import PrimalWeightedUpdate, PrimalRegularization, DualWeightedUpdate
from proximal_operators import ProximalLinfBall


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


class GaussianNoiseGenerator(nn.Module):
    def __init__(self):
        super(GaussianNoiseGenerator, self).__init__()

    def forward(self, img, std, mean=0.0, dtype=torch.cuda.FloatTensor):
        """

        :param img:
        :param std:
        :param mean:
        :param dtype:
        :return:
        """
        noise = torch.zeros(img.size()).type(dtype)
        noise.normal_(mean, std=std)
        img_n = img + noise
        return img_n


class Net(nn.Module):

    def __init__(self, w1, w2, w, max_it, lambda_rof, sigma, tau, theta, dtype=torch.cuda.FloatTensor):
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

        :param x:
        :param img_obs:
        :return:
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


class NetGeneratedNoise(nn.Module):

    def __init__(self, w1, w2, w, max_it=10, lambda_rof=4.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.noise_generator = GaussianNoiseGenerator()
        self.m = nn.ReflectionPad2d(10)
        self.linear_op = LinearOperator()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = lambda_rof

    def forward(self, x, img_obs, dtype=torch.cuda.FloatTensor):
        """
        Forward function. First we generate the parameters of the noise, then we crate the noise to add.
        :param x:
        :param img_ref:
        :return:
        """
        x = Variable(img_obs.data.clone()).type(dtype)
        x_tilde = Variable(x.data.clone()).type(dtype)
        img_obs = Variable(img_obs).type(dtype)
        img_obs.data.type(self.type)
        # Forward pass
        y = self.linear_op(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        self.w.data.type(dtype)
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)

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

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg