
import unittest
import torch
from torch.autograd import Variable

from primal_dual_model import BackwardWeightedDivergence, ForwardWeightedGradient

class TestStringMethods(unittest.TestCase):

    def test_adj(self):
        Y = 200
        X = 100
        x = 1 + Variable(torch.randn((1, Y, X)).type(torch.FloatTensor))
        y_l = Variable(torch.randn((2, Y + 1, X + 1)).type(torch.FloatTensor))

        y_l[0, 1:, 1:-1] = 1 + torch.randn((1, Y, X - 1))
        y_l[1, 1:-1, 1:] = 1 + torch.randn((1, Y - 1, X))
        y = y_l[:, 1:, 1:]
        x = x.cuda()
        y = y.cuda()
        w = Variable(torch.rand(y.size())).cuda()
        # Compute gradient and divergence
        gx = ForwardWeightedGradient().forward(x, w)
        dy = BackwardWeightedDivergence().forward(y, w)

        check = abs((y.data.cpu().numpy()[:] * gx.data.cpu().numpy()[:]).sum() + (dy.data.cpu().numpy()[:] * x.data.cpu().numpy()[:]).sum())
        print(check)
        self.assertTrue(check < 1e-4)

