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
import networkx as nx
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling, to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import random

class GNUMAP2(nn.Module):
    def __init__(self,
                 in_dim,
                 nhid=256,
                 out_dim=2,
                 epochs=400,
                 n_layers=2,
                 fmr=0,
                 gnn_type=None,
                 alpha=None,
                 beta=None):
        super().__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=n_layers, dropout_rate=fmr, gnn_type=gnn_type, alpha=alpha, beta=beta)
        self.epochs, self.in_dim, self.out_dim = epochs, in_dim, out_dim
        self.alpha, self.beta = self.find_ab_params(spread=1, min_dist=0.1)
        
    def find_ab_params(self, spread=1, min_dist=0.1):
        # Exact UMAP function for fitting a, b params
        # spread=1, min_dist=0.1 default umap value -> a=1.58, b=0.9
        # spread=5, min_dist=0.001 -> a=0.15, b=0.79

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    def forward(self, features, edge_index, row_neg=None, col_neg=None):

        # Updates current_embedding, calculates q (probability distribution of node connection in lowdim)

        current_embedding = self.gc(features, edge_index)

        if row_neg is None or col_neg is None:
            q = None
        else:
            # Positive edges denoted by edge_index
            # Negative edges denoted by row_neg, col_neg
            source_embeddings = current_embedding[edge_index[0]]
            target_embeddings = current_embedding[edge_index[1]]
            pos_diff = source_embeddings - target_embeddings
            pos_p_norm_distances = torch.norm(pos_diff, p=2, dim=1)

            source_neg_embeddings = current_embedding[row_neg]
            target_neg_embeddings = current_embedding[col_neg]
            neg_diff = source_neg_embeddings - target_neg_embeddings
            neg_p_norm_distances = torch.norm(neg_diff, p=2, dim=1)
            
            lowdim_dist = torch.cat((pos_diff, neg_diff), dim=0)
            lowdim_dist = torch.norm(lowdim_dist, p=2, dim=1)
            q = 1 / (1 + self.alpha * torch.pow(lowdim_dist, (2*self.beta)))
        return current_embedding, q

    def loss_function(self, p, q, reg):
        def CE(highd, lowd, reg):
            # highd and lowd both have indim x indim dimensions
            #highd, lowd = torch.tensor(highd, requires_grad=True), torch.tensor(lowd, requires_grad=True)
            pos_CE = torch.sum(highd * torch.log(lowd))
            neg_CE = torch.sum((1 - highd) * torch.log(1 - lowd))
            return - (reg* pos_CE + neg_CE)
        
        loss = CE(p, q, reg)
        return loss
        
    def fit(self, features, edge_index, edge_weight, lr=0.005, opt='adam', weight_decay=0):
        loss_values = []
        best = 1e9
        cnt_wait = 0
        print("Starting fit.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


        # Probability distribution in highdim defined as the sparse adj matrix with probability of node connection.
        # No further updates to p


        # More nonzero p than n_neighbors because includes duplicates
        p = torch.zeros((features.shape[0],features.shape[0])) # 1000,1000
        for i in range(len(edge_weight)):
            source = edge_index[0, i]
            target = edge_index[1, i]
            weight = edge_weight[i]
            p[source, target] = weight # create p from edge_index, edge_weight
            # p here is between 0 to 1. That's why it works well with normalized lowdim distances.

    
        # q is probability distribution in lowdim
        # q will be updated at each forward pass
        possible_edges = (features.shape[0]*(features.shape[0]-1))/2
        num_edges = len(edge_index[0])
        reg = possible_edges / num_edges
        pos_p = p[edge_index[0],edge_index[1]]

        self.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            row_neg, col_neg = negative_sampling(edge_index)
            current_embedding, q = self(features, edge_index, row_neg, col_neg)
            q.requires_grad_(True)

            p_sampled = torch.cat((pos_p, p[row_neg, col_neg]), dim=0)
 
            loss = self.loss_function(p_sampled, q, reg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            loss_np = loss.item()
            loss_values.append([loss_np, loss.item(), -1])
            print("Epoch ", epoch, " |  Loss ", loss_np)
            # if round(loss_np, 2) < best:
            #     best = loss
            #     cnt_wait = 0
            # else: 
            #     cnt_wait += 1
            # if cnt_wait == 50 and epoch > 50:
            #     print('Early stopping at epoch {}!'.format(epoch))
            #     break
        return loss_values

    def predict(self, features, edge_index):
        current_embedding, q = self(features, edge_index)
        return current_embedding, q