#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.baseline_models import LogReg, MLP, GCN
from models.data_augmentation import *
from models.dbn import DBN

class GRACE(nn.Module):
    def __init__(self, in_dim, hid_dim, proj_hid_dim, n_layers, tau = 0.5,
                use_mlp = False, dropout_rate = 0.5, gnn_type = "symmetric", alpha = 0.5, beta = 1.0):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, proj_hid_dim, n_layers,dropout_rate, gnn_type = gnn_type, alpha = alpha, beta = beta)
        else:
            self.backbone = MLP(in_dim, hid_dim, proj_hid_dim)

        self.fc1 = nn.Linear(proj_hid_dim, proj_hid_dim)
        self.fc2 = nn.Linear(proj_hid_dim, proj_hid_dim)
        self.fc3 = nn.Linear(proj_hid_dim, proj_hid_dim)
        self.tau = tau


    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        z1 = self.backbone(data1.x, data1.edge_index)
        z2 = self.backbone(data2.x, data2.edge_index)
        return z1, z2

    def projection(self, z, layer="nonlinear-hid"):
        if layer == "nonlinear-hid":
            z = F.elu(self.fc1(z))
            h = self.fc2(z)
        elif layer == "nonlinear":
            h = F.elu(self.fc3(z))
        elif layer == "linear":
            h = self.fc3(z)
        elif layer == "standard":
            h = (z - z.mean(0)) / z.std(0)
        elif layer == 'dbn':
            print(z.shape)
            self.dbn = DBN(num_features=z.shape[1],
                          num_groups=1,
                          dim=2,
                          affine=False, momentum=1.)
            h = self.dbn(z)
        return h

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, layer="nonlinear-hid", mean = True):
        h1 = self.projection(z1, layer)
        h2 = self.projection(z2, layer)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret