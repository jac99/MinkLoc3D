# PointNet code taken from PointNetVLAD Pytorch implementation: https://github.com/cattaneod/PointNetVlad-Pytorch
# Adapted by Jacek Komorowski

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

# NOTE: The toolbox can only pool lists of features of the same length. It was specifically optimized to efficiently
# o so. One way to handle multiple lists of features of variable length is to create, via a data augmentation
# technique, a tensor of shape: 'batch_size'x'max_samples'x'feature_size'. Where 'max_samples' would be the maximum
# number of feature per list. Then for each list, you would fill the tensor with 0 values.


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim, gating=True, add_batch_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        # Expects (batch_size, num_points, channels) tensor
        assert x.dim() == 3
        num_points = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, num_points, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, num_points, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, num_points, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class MinkNetVladWrapper(torch.nn.Module):
    # Wrapper around NetVlad class to process sparse tensors from Minkowski networks
    def __init__(self, feature_size, output_dim, cluster_size=64, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=cluster_size, output_dim=output_dim,
                                     gating=gating, add_batch_norm=True)

    def forward(self, x):
        # x is SparseTensor
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor
