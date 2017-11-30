"""
Project:    pytorch_primal_dual
File:       proximal_operators.py
Created by: louise
On:         29/11/17
At:         3:54 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn


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
