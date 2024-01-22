import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from models.baseline_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import manifold
from sklearn.decomposition import PCA
siglog = torch.nn.LogSigmoid()
import networkx as nx
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling

class GNUMAP2(nn.Module):
    def __init__(self,
                 in_dim,
                 nhid=256,
                 out_dim=2,
                 epochs=500,
                 n_layers=2,
                 fmr=0):
        super().__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=n_layers, dropout_rate=fmr)
        self.epochs, self.in_dim, self.out_dim = epochs, in_dim, out_dim
        self.alpha, self.beta = self.find_ab_params(spread=5, min_dist=0.00001)
        
    def find_ab_params(self, spread=5, min_dist=0.00001):
        """Exact UMAP function for fitting a, b params"""
        # spread=1, min_dist=0.1 default umap value -> a=1.57, b=0.89
        # spread=1, min_dist=0.01 -> a=1.92, b=0.79

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    def forward(self, features, edge_index):
        """
        Updates current_embedding, calculates q (probability distribution of node connection in lowdim)
        """
        current_embedding = self.gc(features, edge_index)
        lowdim_dist = torch.cdist(current_embedding,current_embedding)
        q = 1 / (1 + self.alpha * torch.pow(lowdim_dist, (2*self.beta)))
        return current_embedding, q

    def loss_function(self, p, q):
        def CE(highd, lowd):
            # highd and lowd both have indim x indim dimensions
            #highd, lowd = torch.tensor(highd, requires_grad=True), torch.tensor(lowd, requires_grad=True)
            eps = 1e-9 # To prevent log(0)
            zero_count = torch.sum(torch.eq(highd, 0))
            n_items = highd.numel()
            return - (20*torch.sum(highd * torch.log(lowd + eps)) + torch.sum((1 - highd) * torch.log(1 - lowd + eps)))/1000
            # return - ((n_items/zero_count)*torch.sum(highd * torch.log(lowd + eps)) + torch.sum((1 - highd) * torch.log(1 - lowd + eps)))/n_items
        # loss = CE(torch.triu(p, diagonal=1), torch.triu(q, diagonal=1))
        loss = CE(p,q)
        return loss

    def sampling_loss_function(self, p, q, edge_index):
        def positive(highd, lowd):
            eps = 1e-9  # To prevent log(0)
            return -torch.mean(highd * torch.log(lowd + eps))  # positive term, negative term
        def negative(highd, lowd, row_neg, col_neg):
            eps = 1e-9
            return -torch.mean((1 - highd[row_neg, col_neg]) * torch.log(1 - lowd[row_neg, col_neg] + eps))
        
        # Calculate the loss only for positive edges
        positive_loss = positive(p, q)

        # Calculate the loss only for negative edges
        row_neg, col_neg = negative_sampling(edge_index)
        negative_loss = negative(p, q, row_neg, col_neg)

        # Combine positive and negative losses
        total_loss = positive_loss + negative_loss

        return total_loss

    def density_r(self, array, dist):
        r1 = torch.sum(torch.pow(dist,2)) # sum(edge weight * dist^2) for each row
        r2 = torch.sum(array, axis=1) # sum(edge weights) over each row
        r = torch.log(r1/r2+1e-8) # for stability
        return r

    def fit(self, features, sparse, edge_index, edge_weight, lr=0.005, opt='adam', weight_decay=0, dens_lambda=30000.0):
        loss_values = []
        print("Starting fit.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        """ 
        Probability distribution in highdim defined as the sparse adj matrix with probability of node connection.
        No further updates to p
        """

        # Calculate density term in highdim
        p = torch.tensor(sparse)
        rp = self.density_r(p, torch.cdist(features, features)) # edge weights, distance between datapoints are same in highdim
        """ 
        q is probability distribution in lowdim
        q will be updated at each forward pass
        """

        self.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            current_embedding, q = self(features, edge_index)
            q.requires_grad_(True)
            rq = self.density_r(q, torch.cdist(current_embedding, current_embedding))

            cov_matrix = torch.cov(torch.stack((rp,rq)))
            corr = cov_matrix[0, 1] / (torch.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1]))

            loss = self.loss_function(p, q) #- corr
            #loss = self.sampling_loss_function(p, q, edge_index)
            loss.backward()
            optimizer.step()

            loss_np = loss.item()
            loss_values.append([loss_np, self.loss_function(p, q).item(), corr.item()])
            print("Epoch ", epoch, " |  Loss ", loss_np, " |  Corr ",corr)
        return loss_values, rp.detach().numpy()

    def predict(self, features, edge_index):
        current_embedding, q = self(features, edge_index)
        return current_embedding, q