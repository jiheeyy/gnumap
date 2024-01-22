import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix, to_undirected
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sc
import sklearn as sk
import umap
import pickle
import argparse
#from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
import pandas as pd
import random, time
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.datasets import *

sys.path.append('../')
from models.baseline_models import *
from models.train_models import *
from umap_functions import *
from graph_utils import *
from experiments.create_dataset import *
from experiments.experiment import *
from evaluation_metric import *


tau = 0.5
edr = 0.2
fmr = 0.5
dim = 256
gnn_type = 'symmetric'
alpha = 0.5


for name in ['Cora','Pubmed']:
    results = []
    embeddings = {}
    if name in ['Blob','Circles','Moons','Cora','Pubmed']:
        classification = True
    else: 
        classification = False
    X, y_true, G = data_set(name, n_samples = 500, n_neighbours = 50,features = 'none', standardize = True, 
        centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
        factor = 0.2, noise = 0.05,
        random_state = i, radius = False, epsilon = 0.5, 
        SBMtype = 'lazy')
    new_data = G
    for model_name in ['PCA','LaplacianEigenmap','Isomap','TSNE','UMAP','DenseMAP',
                       'DGI', 'GRACE', 'CCA-SSG', 'Entropy-SSG', 'BGRL',
                       'GNUMAP', 'GNUMAP2','SPAGCN']:
        mod, res, embeds = experiment(model_name, new_data,X,
                            y_true, None,
                            patience=20, epochs=200,
                            n_layers=2, out_dim=2, lr1=1e-3, lr2=1e-2, wd1=0.0,
                            wd2=0.0, tau=tau, lambd=1e-4, min_dist=0.1,
                            method='heat', n_neighbours=15,
                            norm='normalize', edr=edr, fmr=fmr,
                            proj="nonlinear", pred_hid=512, proj_hid_dim=dim,
                            dre1=0.2, dre2=0.2, drf1=0.4, drf2=0.4,
                            npoints = 500, n_neighbors = 50, classification = classification, 
                            densmap = False, random_state = i, n = 15, perplexity = 30, 
                            alpha = alpha, beta = 1.0, gnn_type = gnn_type, 
                            name_file="blob-test",subsampling=None)
        results += [res]
        out = embeds
        embeddings[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha)]  =  {
                                        'model': model_name, 
                                        'alpha': alpha,
                                        'gnn_type': gnn_type,   
                                        'embedding' : out,
                                        'alpha': alpha}
file_path = os.getcwd() + '/' + name + '_traditional_results.csv'

pd.DataFrame(np.array(results),
                columns =[  'model', 'method',
                    'dim', 'neighbours', 'n_layers', 'norm','min_dist',
                        'dre1', 'drf1', 'lr', 'edr', 'fmr',
                    'tau', 'lambd','pred_hid','proj_hid_dim',
                    'sp','acc','local','density','alpha','beta','gnn_type']).to_csv(file_path)


pickle.dump(embeddings, open(os.getcwd() +'/'+name + '_traditional_results.pkl', 'wb'))

print(results)