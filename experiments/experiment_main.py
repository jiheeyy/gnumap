import sys, os
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
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
import traceback
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

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_neighbours', type=int, default=20)
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--out_dim', type=int, default=2)

parser.add_argument('--a', type=float, default=1.)  # data construction
parser.add_argument('--b', type=float, default=1.)  # data construction
parser.add_argument('--radius_knn', type=float, default=0)  # graph construction
parser.add_argument('--bw', type=float, default=1.)  # graph construction
parser.add_argument('--features', type=str, default='lap')  # graph construction

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_img', type=int, default=1)
parser.add_argument('--jm', nargs='+', default=['DGI','BGRL','CCA-SSG','GRACE','GNUMAP2', 'GNUMAP','SPAGCN',
                            'PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP',
                            'GAE','VGAE'],
                    help='List of models to run')
parser.add_argument('--basic', type=int, default=0)
parser.add_argument('--result_file', type=str, default='result_file')
parser.add_argument('--eval', type=int, default=1) # make this 0 to not evaluate
parser.add_argument('--single', type=int, default=0) # make this 1 for single testing
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
save_img = bool(args.save_img)
if args.name_dataset in ['Blobs','Swissroll','Circles','Moons','Sphere']:
    large_class = False
else:
    large_class=True
basic = bool(args.basic)
eval = bool(args.eval)
single = bool(args.single)
if basic:
     models_to_test = ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP']
else:
     models_to_test = args.jm
name_file = args.result_file
new_dir_path = os.path.join(os.getcwd(), 'results/',args.filename)
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)
results = {}

X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name_dataset, n_samples=args.n_samples,features=args.features,featdim = 20,
                                                          n_neighbours=args.n_neighbours,standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise,
                                                          radius_knn=args.radius_knn, bw=args.bw,
                                                          SBMtype='lazy',a=args.a,
                                                          b=args.b, random_state=seed)

def visualize_dataset(X_ambient, cluster_labels, title, save_img, save_path):
    if save_img:
        plt.figure(figsize=(4, 4))
        if title[:5] == 'Mouse':
            color_palette = ["#877688", "#73377f", "#1c9a70", "#35609f"]
            gray_color = "#877688"

            cluster_to_color = {cluster: color_palette[i] for i, cluster in enumerate(sorted(cluster_labels.unique()))}
            mapped_colors = cluster_labels.map(cluster_to_color).values
            is_gray = mapped_colors == gray_color

            plt.scatter(X_ambient[is_gray, 0], X_ambient[is_gray, 1], s=1, c=gray_color, alpha=0.2)
            plt.scatter(X_ambient[~is_gray, 0], X_ambient[~is_gray, 1], s=1, c=mapped_colors[~is_gray])
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(False)
        else:
            plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=cluster_labels, s=8, cmap=plt.cm.Spectral)
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(False)
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
    else:
        pass

visualize_dataset(X_manifold, cluster_labels, 
title=f'{args.name_dataset} a={str(args.a)}, b={str(args.b)}, {str(args.n_samples)} Points', 
save_img=save_img, save_path=new_dir_path + "/manifold_" + args.name_dataset + ".png")

def visualize_density(X_ambient, rp, title, model_name, file_name, save_path):
    if rp is not None:
        plt.figure (figsize=(8, 6))
        plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=rp, cmap=plt.cm.viridis)
        final_save_path = os.path.join(save_path, 'DENSITY.png')
        plt.savefig(final_save_path, format='png', dpi=300)
        plt.close()
    else:
        pass


def visualize_embeds(X, loss_values, cluster_labels, title, model_name, file_name, save_path):
    fig, (ax1) = plt.subplots(1, 1, figsize=(4,4))
    if X is not None:
        if X.shape[1] == 3:
            # 3D scatter plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap=plt.cm.Spectral)
        elif X.shape[1] == 2:
            if args.name_dataset[:5] == 'Mouse':
                color_palette = ["#877688", "#73377f", "#1c9a70", "#35609f"]
                gray_color = "#877688"
                cluster_to_color = {cluster: color_palette[i] for i, cluster in enumerate(sorted(cluster_labels.unique()))}
                mapped_colors = cluster_labels.map(cluster_to_color).values
                is_gray = mapped_colors == gray_color
                ax1.scatter(X[is_gray, 0], X[is_gray, 1], s=1, c=gray_color, alpha=0.2)
                ax1.scatter(X[~is_gray, 0], X[~is_gray, 1], s=1, c=mapped_colors[~is_gray])
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().get_xaxis().set_visible(False)
            else:
                # 2D scatter plot
                ax1.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap=plt.cm.Spectral, s=8)
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().get_xaxis().set_visible(False)

        # Output dimension more than 3
        else:
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')

        # ax1.set_title(title)
    else:
        # If X is None, set the background to black
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')

    # Plotting loss values on the second subplot
    # if model_name not in ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP']:
    #     ax2.plot(loss_values, color='blue')
    #     ax2.set_title('Loss Over Time')
    #     ax2.set_xlabel('Epoch')
    #     ax2.set_ylabel('Loss')
    # else:
    #     pass
    
    final_save_path = os.path.join(save_path, model_name+file_name+'.png')
    plt.savefig(final_save_path, format='png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()

alpha_array = [0.5]
beta_array = [1]
lambda_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
tau_array = [0.1, 0.2, 0.5, 1., 10]
type_array = ['symmetric']
fmr_array = [0, 0.2, 0.5]
edr_array = [0, 0.2, 0.5]
spagcn_n_neighbors=[5, 10, 20]
spagcn_res=[0.3, 0.4, 0.5]
spagcn_alpha = [0.1,0.2,0.3]

hyperparameters = {
    'DGI': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'fmr':fmr_array},
    'CCA-SSG': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'lambd':lambda_array, 'fmr':fmr_array, 'edr':edr_array},
    'BGRL': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'lambd':lambda_array, 'fmr':fmr_array, 'edr':edr_array},
    'GRACE': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'tau':tau_array, 'fmr':fmr_array, 'edr':edr_array},
    'GNUMAP2':{'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array,'fmr':fmr_array},
    'SPAGCN': {'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array,'fmr':fmr_array, 
    'spagcn_n_neighbors':spagcn_n_neighbors, 'spagcn_res':spagcn_res, 'spagcn_alpha':spagcn_alpha},
    'PCA':{}, 'LaplacianEigenmap':{}, 'Isomap':{}, 'TSNE':{}, 'UMAP':{}, 'DenseMAP':{},
    'VGAE':{'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'fmr':fmr_array},
    'GAE':{'alpha':alpha_array, 'beta':beta_array, 'gnn_type':type_array, 'fmr':fmr_array}
}
args_params = {
    'epochs': args.epoch,
    'n_layers': args.n_layers,
    'hid_dim': args.hid_dim,
    'lr': args.lr,
    'n_neighbors': args.n_neighbours,
    'dataset': args.name_dataset,
    'save_img': args.save_img,
}


for model_name in models_to_test:
    best_acc = 0
    if model_name not in ['DGI','BGRL','GRACE','CCA-SSG','GNUMAP2', 'GNUMAP','SPAGCN',
                            'PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE', 'UMAP', 'DenseMAP',
                            'VGAE','GAE']:
                            raise ValueError('Invalid model name')
    model_hyperparameters = hyperparameters[model_name]
    if args.out_dim:
        out_dim = args.out_dim
    else:
        out_dim=min(X_manifold.shape[1],2)

    for combination in product(*model_hyperparameters.values()):
        params = dict(zip(model_hyperparameters.keys(), combination))

        try:
            mod, res, out, loss_values, rp = experiment(model_name, G, X_ambient, X_manifold, cluster_labels, large_class,
                        out_dim=out_dim, name_file=name_file, 
                        random_state=42, perplexity=30, wd=0.0, pred_hid=512,proj="standard",min_dist=1e-3,patience=20,
                        eval=eval,
                        **args_params,
                        **params)
            res['save_img'] = False
            if save_img and res['acc'] > best_acc:
                best_acc = res['acc']
                res['save_img'] = True # most recent True means that image was saved for the model
                visualize_embeds(out, loss_values, cluster_labels, f"{model_name}, {params}", model_name, str(args_params)+str(args.features),
                new_dir_path) 
            else:
                pass
        except:
            res = None
            print(traceback.format_exc())
        results[model_name+ '_' + name_file + str(params)] = res if res is not None else {}
        if single:
            break

file_path = new_dir_path + '/' + args.name_dataset + '_' + str(args.seed) + '.csv'
pd.DataFrame.from_dict(results, orient='index').to_csv(file_path)
