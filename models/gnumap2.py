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
from torch_geometric.utils import negative_sampling, to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

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
        self.alpha, self.beta = self.find_ab_params(spread=5, min_dist=0.001)
        
    def find_ab_params(self, spread=1, min_dist=0.1):
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

    # def forward(self, features, edge_index):
    #     """
    #     Updates current_embedding, calculates q (probability distribution of node connection in lowdim)
    #     """
    #     row_neg, col_neg = negative_sampling(edge_index)
    #     # Positive edges denoted by edge_index
    #     # Negative edges denoted by row_neg, col_neg

    #     current_embedding = self.gc(features, edge_index)

    #     # I want to make these two lines faster by sampling only from edge_index and row_neg, col_neg
    #     lowdim_dist = torch.cdist(current_embedding,current_embedding)
    #     q = 1 / (1 + self.alpha * torch.pow(lowdim_dist, (2*self.beta)))
    #     return current_embedding, q
    def forward(self, features, edge_index, row_neg=None, col_neg=None):
        """
        Updates current_embedding, calculates q (probability distribution of node connection in lowdim)
        """
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
            
            lowdim_dist = torch.cat((pos_p_norm_distances, neg_p_norm_distances), dim=0)
            q = 1 / (1 + self.alpha * torch.pow(lowdim_dist, (2*self.beta)))
        return current_embedding, q

    def loss_function(self, p, q,reg):
        
        def CE(highd, lowd, reg):
            # highd and lowd both have indim x indim dimensions
            #highd, lowd = torch.tensor(highd, requires_grad=True), torch.tensor(lowd, requires_grad=True)
            eps = 1e-9 # To prevent log(0)
            return - (reg * torch.sum(highd * torch.log(lowd + eps)) + \
                torch.sum((1 - highd) * torch.log(1 - lowd + eps))) / len(highd)
        
        loss = CE(p, q, reg)
        return loss

    def density_r(self, array, dist):
        r1 = torch.sum(torch.pow(torch.tensor(dist),2)) # sum(edge weight * dist^2) for each row
        r2 = torch.sum(array, axis=1) # sum(edge weights) over each row
        r = torch.log(r1/r2+1e-8) # for stability
        return r

    def fit(self, features, edge_index, edge_weight, lr=0.005, opt='adam', weight_decay=0):
        loss_values = []
        best = 1e9
        cnt_wait = 0
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
        p = torch.zeros((features.shape[0],features.shape[0])) # 1000,1000
        for i in range(len(edge_weight)):
            source = edge_index[0, i]
            target = edge_index[1, i]
            weight = edge_weight[i]
            p[source, target] = weight # create p from edge_index, edge_weight

        # sparse_p = to_scipy_sparse_matrix(edge_index, num_nodes=features.shape[0])
        # pdist = shortest_path(csr_matrix(sparse_p),directed = False)
        # pdist[np.isinf(pdist)] = np.max(pdist[~np.isinf(pdist)])*2

        #rp = self.density_r(p, pdist) # edge weights, distance between datapoints are same in highdim
        """ 
        q is probability distribution in lowdim
        q will be updated at each forward pass
        """

        # Calculate loss regularizer
        n_items = p.shape[0] ** 2
        pos_edge_count = edge_index.shape[1]
        neg_edge_count = n_items - pos_edge_count
        reg = neg_edge_count / pos_edge_count

        self.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            row_neg, col_neg = negative_sampling(edge_index)
            current_embedding, q = self(features, edge_index, row_neg, col_neg)
            q.requires_grad_(True)
            # rq = self.density_r(q, torch.cdist(current_embedding, current_embedding))

            # cov_matrix = torch.cov(torch.stack((rp,rq)))
            # corr = cov_matrix[0, 1] / torch.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
            p_sampled = torch.cat((p[edge_index[0],edge_index[1]], p[row_neg, col_neg]), dim=0)
            loss = self.loss_function(p_sampled, q, reg)
            loss.backward()
            optimizer.step()
            loss_np = loss.item()
            loss_values.append([loss_np, loss.item(), -1]) #corr.item()
            print("Epoch ", epoch, " |  Loss ", loss_np) # Corr ",corr
            print(reg)

            # if round(loss_np, 2) < best:
            #     best = loss
            #     cnt_wait = 0
            # else: 
            #     cnt_wait += 1
            # if cnt_wait == 50 and epoch > 50:
            #     print('Early stopping at epoch {}!'.format(epoch))
            #     break
        return loss_values #rp.detach().numpy()

    def predict(self, features, edge_index):
        current_embedding, q = self(features, edge_index)
        return current_embedding, q