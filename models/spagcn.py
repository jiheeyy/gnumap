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


# # INPUT 1: sparse matrix, numpy array form
# A_dist = kneighbors_graph(X, N_NEIGHBOURS, mode='distance', include_self=False) # adjacency matrix
# edge_index, edge_weights = from_scipy_sparse_matrix(A_dist)
# edge_index, edge_weights = to_undirected(edge_index, edge_weights)

# # INPUT 2: input features: identity matrix
# A = torch.eye(X.shape[0])


class SPAGCN(nn.Module):
    def __init__(self,
                in_dim,
                hid_dim,
                out_dim,
                epochs,
                n_layers,
                fmr=0,
                n_clusters=10, #kmeans
                spagcn_alpha=0.5):
        super(SPAGCN, self).__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, n_layers=n_layers, dropout_rate=fmr)
        self.n_clusters = n_clusters  # for fit method kmeans
        self.mu = Parameter(torch.Tensor(n_clusters, out_dim))
        self.alpha = spagcn_alpha
        self.epochs = epochs
        self.out_dim = out_dim

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=self.out_dim) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, x, adj,
            lr=0.005,
            update_interval=3,
            weight_decay=0,
            opt="adam",
            init="kmeans",
            n_neighbors=10,
            res=0.4,
            init_spa=True,
            tol=1e-3):
        max_epochs=self.epochs
        loss_values = []
        print("Starting fit.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(x, adj)

        if init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # ------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # ------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  # Here we use X as numpy
        elif init == "louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata = sc.AnnData(features.detach().numpy())
            else:
                adata = sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        # ----------------------------------------------------------------
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(x, adj)
                p = self.target_distribution(q).data
            optimizer.zero_grad()
            z, q = self(x, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            loss_numpy = loss.detach().numpy()
            loss_values.append(loss_numpy)
            print("Epoch ", epoch, "Loss ", loss_numpy)
        return loss_values

    def predict(self, x, adj):
        z, q = self(torch.Tensor(x), torch.Tensor(adj))
        return z, q