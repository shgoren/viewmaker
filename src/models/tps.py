# adapted from https://github.com/WarBean/tps_stn_pytorch


import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class TPSDecoder(nn.Module):

    def __init__(self, input_dim, image_hw, span_range_hw=(0.9, 0.9), grid_hw=(4,4)):
        """
        :param input_dim: (C,H,W)
        :param image_hw: original image dimensions (H,W)
        :param span_range_hw:
        :param grid_hw: number of points of control points (H,W)
        """
        super(TPSDecoder, self).__init__()
        self.input_dim = input_dim
        self.span_range_hw = span_range_hw
        self.grid_hw = grid_hw
        self.image_hw = image_hw

        r1, r2 = span_range_hw

        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_hw[0] - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_hw[1] - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.target_control_points = torch.cat([X, Y], dim=1)

        self.loc_net = GridLocNet(self.input_dim, self.grid_hw, self.target_control_points)

        self.tps = TPSGridGen(*image_hw, self.target_control_points)

    def forward(self, f, img, budget):
        source_control_points = self.loc_net(f)
        budgeted_control_points = self.restrain_control_points(source_control_points, budget)
        source_coordinate = self.tps(budgeted_control_points)
        grid = source_coordinate.view(-1, *self.image_hw, 2)
        transformed_x = grid_sample(img, grid)
        return transformed_x

    def restrain_control_points(self, source_control_points, budget):
        self.target_control_points = self.target_control_points.to(source_control_points.device)
        diff_vectors = self.target_control_points - source_control_points
        diff_norm = torch.norm(diff_vectors, dim=2)
        morph_size = diff_norm.mean(1)
        delta = diff_vectors/(morph_size.reshape(-1,1,1) + 1e-5) * budget
        restrained_control_points = self.target_control_points - delta
        return restrained_control_points


class GridLocNet(nn.Module):
    def __init__(self, input_dim, grid_hw, target_control_points, bounded=False, hidden_dim=50, squeeze_dim=32):
        """

        :param input_dim: (C,H,W)
        :param grid_height: number of points of control points (H,W)
        :param target_control_points:
        :param bounded: allow control points outside of image
        :param hidden_dim: hidden dim for the control point prediction Perceptron
        """
        super(GridLocNet, self).__init__()

        self.bounded = bounded
        self.grid_height, self.grid_width = grid_hw
        self.squeeze = nn.Conv2d(input_dim[0], squeeze_dim, 1)
        self.fc_input_size = squeeze_dim * input_dim[1] * input_dim[2]
        self.fc1 = nn.Linear(self.fc_input_size, hidden_dim)
        # TODO: replace with target_control_points.view(-1).size(0)
        self.fc2 = nn.Linear(hidden_dim, self.grid_height * self.grid_width * 2)

        if self.bounded:
            bias = torch.atanh(target_control_points).view(-1)
        else:
            bias = target_control_points.view(-1)
        self.fc2.bias.data.copy_(bias)
        self.fc2.weight.data = self.fc2.weight * 1e-4

    def forward(self, f):
        batch_size = f.size(0)
        f = self.squeeze(f).view(-1, self.fc_input_size)
        f = F.relu(self.fc1(f))
        f = F.dropout(f, training=self.training)
        f = self.fc2(f)
        if self.bounded:
            points = F.tanh(f)
        else:
            points = f
        return points.view(batch_size, -1, 2)


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid, mode="nearest")
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def test_tps():
    import matplotlib.pyplot as plt
    tps = TPSDecoder(64, (32,32))
    f = torch.rand((1,64))
    img = torch.rand((1,1,32,32))
    res = tps(f, img, 0.05)
    print(res)
