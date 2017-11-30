"""
Project:    pytorch_primal_dual
File:       differential_operators.py
Created by: louise
On:         29/11/17
At:         3:53 PM
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