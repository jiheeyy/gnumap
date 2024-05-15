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
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SPAGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, epochs, fmr, gnn_type, alpha, beta):
        super(SPAGCN, self).__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, n_layers=1, dropout_rate=fmr,
        alpha=0.5, beta=0.5, gnn_type=gnn_type)
        self.epochs = epochs
        self.out_dim=out_dim
        #self.mu determined by the init method
        self.alpha=alpha

    def forward(self, x, edge_index, edge_weight):
        x=self.gc(x, edge_index, edge_weight)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, edge_index, edge_weight, lr=0.005, update_interval=3, weight_decay=0,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-3):
        max_epochs=self.epochs
        loss_values=[]

        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        features= self.gc(X, edge_index, edge_weight)
        #----------------------------------------------------------------        
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
        elif init=="louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(features.detach().numpy())
            else:
                adata=sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.out_dim))
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(X, edge_index, edge_weight)
                p = self.target_distribution(q).data
            if epoch%10==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, edge_index, edge_weight)
            loss = self.loss_function(p, q)          
            loss.backward()
            optimizer.step()
            loss_numpy = loss.detach().numpy()
            loss_values.append(loss_numpy)
            print("Epoch ", epoch, "Loss ", loss_numpy)

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
        return loss_values

    def predict(self, X, edge_index, edge_weight):
        z,q = self(X, edge_index, edge_weight=edge_weight)
        return z, q