import sys, os
import logging
logging.basicConfig(filename='expmain.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx, numpy as np
from numbers import Number
import math
import pandas as pd
import random, time
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from itertools import product
from sklearn.datasets import *

sys.path.append('../')
from models.baseline_models import *
from models.train_models import *
from gnumap.umap_functions import *
from graph_utils import *
from experiments.create_dataset import *
from experiments.experiment import *
from metrics.evaluation_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--name_dataset', type=str, default='Swissroll')
parser.add_argument('--filename', type=str, default='test')
parser.add_argument('--split', type=str, default='PublicSplit')

parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_neighbours', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--hid_dim', type=int, default=128)

parser.add_argument('--a', type=float, default=1.)  # data construction
parser.add_argument('--b', type=float, default=1.)  # data construction
parser.add_argument('--radius_knn', type=float, default=0)  # graph construction
parser.add_argument('--bw', type=float, default=1.)  # graph construction

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_img', type=bool, default=True)
parser.add_argument('--jcsv', type=float, default=False)  # make csv?
parser.add_argument('--jm', nargs='+', default=['SPAGCN', 'DGI','GRACE', 'CCA-SSG'],
                    help='List of models to run')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
save_img = args.save_img
name_file = args.name_dataset + "_" + args.filename
results = {}
logging.info('STARTING EXPERIMENT')

X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name_dataset, n_samples=1000,features='none',featdim = 20,
                                                          n_neighbours=args.n_neighbours,standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise,
                                                          radius_knn=args.radius_knn, bw=args.bw,
                                                          SBMtype='lazy',a=args.a,
                                                          b=args.b)

def visualize_dataset(X_ambient, cluster_labels, title, save_img, save_path):
    if save_img:
        plt.figure()
        plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)
        plt.title(title)
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        pass

visualize_dataset(X_manifold, cluster_labels, title=args.name_dataset, save_img=save_img,
                  save_path=os.getcwd() + '/results/' + "gt_manifold_" + name_file + ".png")
visualize_dataset(X_ambient, cluster_labels, title=args.name_dataset, save_img=save_img,
                  save_path=os.getcwd() + '/results/' + "gt_ambient_" + name_file + ".png")


def visualize_embeds(X, loss_values, cluster_labels, title, model_name, file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if X is not None:
        if model_name == 'GRACE':
            # 3D scatter plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap=plt.cm.Spectral)
        else:
            # 2D scatter plot
            ax1.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)

        ax1.set_title(title)
    else:
        # If X is None, set the background to black
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')

    # Plotting loss values on the second subplot
    if model_name not in ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP']:
        ax2.plot(loss_values, color='blue')
        ax2.set_title('Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
    else:
        pass
    
    save_path = os.path.join(os.getcwd(), 'results', model_name+file_name+'.png')
    plt.savefig(save_path, format='png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()

alpha_array = [0.5] #np.arange(0,1,0.5)
beta_array = [0.5] #np.arange(0,1,0.5)
lambda_array = [1e-3] #[1e-4, 1e-3, 1e-2, 1e-1, 1.]
tau_array = [0.5] #[0.1, 0.2, 0.5, 1., 10]
type_array = ['symmetric'] #['symmetric','RW']
fmr_array = [0.1] #[0.1,0.6]
edr_array = [0] #[0,0.1]

hyperparameters = {
    'DGI': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array},
    'GNUMAP':{'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array},
    'CCA-SSG': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'lambd':lambda_array, 'fmr':fmr_array, 'edr':edr_array},
    'BGRL': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'lambd':lambda_array, 'fmr':fmr_array, 'edr':edr_array},
    'GRACE': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'tau':tau_array, 'fmr':fmr_array, 'edr':edr_array},
    'SPAGCN':{},
    'PCA':{}, 'LaplacianEigenmap':{}, 'Isomap':{}, 'TSNE':{}, 'UMAP':{}, 'DenseMAP':{}
}
args_params = {
    'epochs': args.epoch,
    'n_layers': args.n_layers,
    'hid_dim': args.hid_dim,
    'lr': args.lr,
    'n_neighbors': args.n_neighbours,
    'dataset': args.name_dataset,
    'save_img': args.save_img
}


for model_name in args.jm:
    if model_name not in ['DGI','BGRL','GRACE','CCA-SSG','SPAGCN', 'GNUMAP',
                            'PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP']:
                            raise ValueError('Invalid model name')
    model_hyperparameters = hyperparameters[model_name]
    if model_name =='GRACE':
        out_dim=X_ambient.shape[1]
    else:
        out_dim=X_manifold.shape[1]

    for combination in product(*model_hyperparameters.values()):
        params = dict(zip(model_hyperparameters.keys(), combination))

        mod, res, out, loss_values = experiment(model_name, G, X_ambient, X_manifold, cluster_labels, 
                    out_dim=out_dim, name_file=name_file, 
                    random_state=42, perplexity=30, wd=0.0, pred_hid=512,proj="standard",min_dist=1e-3,patience=20,
                    **args_params,
                    **params)
        if save_img:
            visualize_embeds(out, loss_values, cluster_labels, f"{model_name}, {params}", model_name, str(args_params)) 
        else:
            pass
        
        logging.info(name_file+str(params))
        results[model_name+ '_' + name_file + str(params)] = res if res is not None else {}

if args.jcsv:
    file_path = os.getcwd() + '/results/' + 'gnn_results_' + str(args.seed) + '_' + name_file + '.csv'
    pd.DataFrame.from_dict(results, orient='index').to_csv(file_path)

logging.info('ENDING EXPERIMENT')
