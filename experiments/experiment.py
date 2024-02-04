import csv
import argparse
from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx, numpy as np
from numbers import Number
import math
import pandas as pd
import random, time
import sys, os

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
import umap

sys.path.append('../')
from data_utils import *
from metrics.edge_prediction import edge_prediction
from graph_utils import get_weights, transform_edge_weights
from metrics.label_classification import label_classification
from models.baseline_models import *
from models.train_models import *
from metrics.evaluation_metrics import *
from gnumap.umap_functions import *
from models.gnumap2 import *
from experiments.create_dataset import *


def experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                epochs=np.nan, n_layers=np.nan, out_dim=np.nan, hid_dim=np.nan, 
                lr=np.nan, n_neighbors=np.nan, dataset=np.nan,
                alpha=np.nan, beta=np.nan, gnn_type=np.nan, tau=np.nan, lambd=np.nan, edr=np.nan, fmr=np.nan,
                name_file=np.nan, save_img=np.nan,
                random_state=42, perplexity=30, wd=0.0, pred_hid=512,proj="standard",min_dist=1e-3,patience=20):
    # num_classes = int(data.y.max().item()) + 1
    loss_values = [1] # placeholder for 6 models without training
    rp = None
    start_time = time.time()

    if model_name == 'DGI':  # a b type
        model, loss_values = train_dgi(G, hid_dim=hid_dim, out_dim=out_dim,
                                       n_layers=n_layers,
                                       patience=patience,
                                       epochs=epochs, lr=lr,
                                       name_file=name_file,
                                       alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'GRACE':  # a b type t
        model, loss_values = train_grace(G, channels=hid_dim, proj_hid_dim=out_dim,
                                         tau=tau,
                                         epochs=epochs, lr=lr, wd=wd,
                                         fmr=fmr, edr=edr, proj=proj, name_file=name_file,
                                         alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'CCA-SSG':  # a b type lam
        model, loss_values = train_cca_ssg(G, hid_dim=hid_dim,
                                           channels=out_dim,
                                           lambd=lambd,
                                           n_layers=n_layers,
                                           epochs=epochs, lr=lr,
                                           fmr=fmr, edr=edr, name_file=name_file,
                                           alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)
    elif model_name == 'Entropy-SSG':
        model, loss_values = train_entropy_ssg(G, hid_dim=hid_dim,
                                               channels=out_dim,
                                               lambd=lambd,
                                               n_layers=n_layers,
                                               epochs=epochs, lr=lr,
                                               fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(G)

    elif model_name == 'BGRL':  # lamb a b type
        model, loss_values = train_bgrl(G, hid_dim, out_dim,
                                        lambd=lambd,
                                        n_layers=n_layers,
                                        epochs=epochs, lr=lr,
                                        fmr=fmr, edr=edr,
                                        pred_hid=pred_hid, wd=wd,
                                        drf1=fmr, drf2=fmr, dre1=edr,
                                        dre2=edr, name_file=name_file,
                                        alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)
    elif model_name == "GNUMAP":  # alpha beta type
        model, embeds, loss_values = train_gnumap(G, hid_dim, out_dim,
                                                  n_layers=n_layers,
                                                  epochs=epochs, lr=lr, wd=wd, name_file=name_file,
                                                  alpha=alpha, beta=beta, gnn_type=gnn_type)
    elif model_name == "GNUMAP2":
        model, embeds, loss_values = train_gnumap2(G, hid_dim, out_dim, epochs, n_layers, fmr)
    elif model_name == "SPAGCN":  # alpha TODO
        model, embeds, loss_values = train_spagcn(G, hid_dim, out_dim, epochs, fmr)
    elif model_name == 'PCA':
        model = PCA(n_components=out_dim)
        embeds = model.fit_transform(
            X_ambient)  # StandardScaler().fit_transform(X) --already standardized when converting graphs
    elif model_name == 'LaplacianEigenmap':
        model = manifold.SpectralEmbedding(n_components=out_dim, n_neighbors=n_neighbors)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'Isomap':
        model = manifold.Isomap(n_components=out_dim)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'TSNE':
        model = manifold.TSNE(n_components=out_dim, learning_rate='auto',
                              init='random', perplexity=perplexity)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'UMAP':
        model = umap.UMAP(n_components=out_dim, random_state=random_state,
                          n_neighbors=n_neighbors, min_dist=min_dist)
        embeds = model.fit_transform(X_ambient)

    elif model_name == 'DenseMAP':
        model = umap.UMAP(n_components=out_dim, random_state=random_state,
                          densmap=True, n_neighbors=n_neighbors,
                          min_dist=min_dist)
        embeds = model.fit_transform(X_ambient)
    else:
        raise ValueError("Model unknown!!")

    try:
        loss_values = [item.item() for item in loss_values]
    except:
        pass
    
    end_time = time.time()
    if model_name == 'GNUMAP2':
        if np.isnan(loss_values[-1][:2]).any():
            embeds = None
            results = None
        else:
            global_metrics, local_metrics = eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,model_name,
                                                     dataset=dataset)
            print("done with the embedding evaluation")
            results=[]
            results = {**global_metrics, **local_metrics}
            results['model_name'] = model_name
            results['out_dim'] = out_dim
            results['hid_dim'] = hid_dim
            results['n_neighbors'] = n_neighbors
            results['min_dist'] = min_dist
            results['lr'] = lr
            results['edr'] = edr
            results['fmr'] = fmr
            results['tau'] = tau
            results['lambd'] = lambd
            results['pred_hid'] = pred_hid
            results['alpha_gnn'] = alpha
            results['beta_gnn'] = beta
            results['gnn_type'] = gnn_type
            results['time'] = end_time - start_time
    elif np.isnan(loss_values[-1]).any():
        embeds = None
        results = None
    else:
        global_metrics, local_metrics = eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,model_name,
                                                 dataset=dataset)
        print("done with the embedding evaluation")
        results=[]
        results = {**global_metrics, **local_metrics}
        results['model_name'] = model_name
        results['out_dim'] = out_dim
        results['hid_dim'] = hid_dim
        results['n_neighbors'] = n_neighbors
        results['min_dist'] = min_dist
        results['lr'] = lr
        results['edr'] = edr
        results['fmr'] = fmr
        results['tau'] = tau
        results['lambd'] = lambd
        results['pred_hid'] = pred_hid
        results['alpha_gnn'] = alpha
        results['beta_gnn'] = beta
        results['gnn_type'] = gnn_type
        results['time'] = end_time - start_time

    return (model, results, embeds, loss_values, rp)