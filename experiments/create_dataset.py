import argparse
import copy, collections
import numpy as np
from numbers import Number
import math
import sys, os

import torch_geometric.transforms as T
from torch_geometric.data import Data
import sklearn.datasets as datasets
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

sys.path.append('../')
from data_utils import *
from graph_utils import convert_to_graph, mouse_convert_to_graph
from experiments.SBM.read_SBM import *
from experiments.simulation_utils import *
import networkx as nx
import matplotlib.pyplot as plt
from ogb.nodeproppred import PygNodePropPredDataset

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from sklearn.preprocessing import StandardScaler
import joblib
from io import BytesIO
import requests
import scanpy as sp

import os.path as osp
from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce

def create_dataset(name, n_samples = 500, n_neighbours = 50, features='none',featdim = 50,
                   standardize=True, centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
                   ratio_circles = 0.2, noise = 0.05, 
                   a=1, b=1, n_bins = 10, random_state = None, radius_knn = 0, bw = 1,
                   SBMtype = 'lazy', nb_loops=5, radius_tube=4, radius_torus=10):

    if name == 'Blobs':
        X_ambient, cluster_labels = datasets.make_blobs( n_samples=n_samples, centers=centers, 
                                        cluster_std=cluster_std, random_state=random_state)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)
        X_manifold = X_ambient

    elif name == 'Sphere':
        X_ambient, X_manifold, cluster_labels   = create_sphere(r = 1, size = n_samples,  a = a, b=b, n_bins = n_bins)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)

    elif name == 'Circles':
        X_ambient, X_manifold, cluster_labels =create_circles(ratio = ratio_circles, size = n_samples, a = a, b=b, 
                                                              noise= noise, n_bins = n_bins)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                            radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)

    elif name == 'Moons':
        X_ambient, X_manifold, cluster_labels = create_moons(size = n_samples, a = a, b=b, noise= noise, n_bins = n_bins)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)

    elif name == 'Swissroll':
        X_ambient, X_manifold, cluster_labels = create_swissroll(size = n_samples, a = a, b=b, noise= noise, n_bins = n_bins)
        G = convert_to_graph(X_manifold, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)

    elif name == "Trefoil":
        X_ambient, X_manifold, cluster_labels = create_trefoil(size = n_samples, a = a, b=b, noise= noise, n_bins = n_bins)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)
        
    elif name == "Helix":
        X_ambient, X_manifold, cluster_labels = create_helix(size = n_samples, a = a, b=b, 
                                                             noise= noise, n_bins = n_bins,
                                                             radius_torus=radius_torus, 
                                                             radius_tube=radius_tube, nb_loops=nb_loops)
        G = convert_to_graph(X_ambient, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw,featdim = featdim,)
        G.y = torch.from_numpy(cluster_labels)
             
    elif name == 'SBM':
        ### needs to be automated
        X_ambiant, cluster_labels, G = readSBM(type = SBMtype, features = features)
        X_manifold = X_ambiant 

    elif name == 'Cora':
        dataset = Planetoid(root='Planetoid', name='Cora', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X_ambient, cluster_labels = G.x.numpy(), G.y.numpy()
        X_manifold = X_ambient

    elif name == 'Pubmed':
        dataset = Planetoid(root='Planetoid', name='Pubmed', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X_ambient, cluster_labels = G.x.numpy(), G.y.numpy()
        X_manifold = X_ambient

    elif name == 'Citeseer':
        dataset = Planetoid(root='Planetoid', name='Citeseer', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X_ambient, cluster_labels = G.x.numpy(), G.y.numpy()
        X_manifold = X_ambient
    
    elif name == 'Products':
        dataset = PygNodePropPredDataset(name = 'ogbn-products', transform=NormalizeFeatures()) 
        G = dataset[0]
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X_ambient, cluster_labels = G.x.numpy(), G.y.numpy()
        X_manifold = X_ambient

    elif name == 'Cancer':
        adata = sc.read("cancer/sample_data.h5ad")
        adj=np.loadtxt('cancer/adj.csv', delimiter=',')
        adata.var_names_make_unique()
        spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        #Normalize and take log for UMI
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        G.x = adata.X.toarray()
        G.edge_weight = torch.exp((edge_weights - max(edge_weights))/max(edge_weights))
    
    elif name == 'Mouse1' or  name == 'Mouse2' or name =='Mouse3':
        spatial_lda_models = {}  

        PATH_TO_3MODEL = "spleen/spleen_training_penalty=0.25_topics=3_trainfrac=0.99.pkl"
        URL_SPLEEN_DF_PKL = "https://github.com/calico/spatial_lda/raw/primary/data/spleen/spleen_df.pkl"
        PATH_TO_SPLEEN_DF_PKL = "spleen/spleen_dfs.pkl" 
        PATH_TO_SPLEEN_FEATURES_PKL = "spleen/spleen_cells_features.pkl" 

        
        spatial_lda_models[3] = pickle.load(open(PATH_TO_3MODEL, "rb"))

        try:
            codex_df_dict = pickle.load(open(PATH_TO_SPLEEN_DF_PKL, "rb"))
        except:
            codex_df_dict = joblib.load(BytesIO(requests.get(URL_SPLEEN_DF_PKL).content))
            with open(PATH_TO_SPLEEN_DF_PKL, "wb") as file:
                pickle.dump(codex_df_dict, file)

        for df in codex_df_dict.values():
            df['x'] = df['sample.X']
            df['y'] = df['sample.Y']
        wt_samples = [ x for x in codex_df_dict.keys() if x.startswith("BALBc")]
        spleen_dfs = dict(zip(wt_samples, [ codex_df_dict[x] for x in wt_samples]))
  
        with open(PATH_TO_SPLEEN_FEATURES_PKL, 'rb') as f:
            spleen_cells_features = pickle.load(f)

        graph_list, cluster_labels, coord_list = mouse_convert_to_graph(spleen_dfs,'sample.X', 'sample.Y', 
        spatial_lda_models[3].topic_weights, sample=True)
        # z_col=None, n_neighbours = 0, features='markers', processing ='znorm', 
        # radius_knn = 100, bw = None)

        num = name[-1]
        G = graph_list[f'BALBc-{num}']
        X_ambient = coord_list[f'BALBc-{num}']
        X_manifold = X_ambient
        cluster_labels = cluster_labels[f'BALBc-{num}']

    elif name.split('_')[0] == 'road':
        class Roads(InMemoryDataset):
            r"""The Roads dataset from SSTD 2005 "On Trip Planning Queries in Spatial Databases"
            https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm

            Args:
                root (str): Root directory where the dataset should be saved.
                name (str): The name of the dataset (:obj:`"cal"`, :obj:`"SF"`, 
                    :obj:`"NA"`, :obj:`"TG"`, :obj:`"OL"`).
                transform (callable, optional): A function/transform that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a transformed
                    version. The data object will be transformed before every access.
                    (default: :obj:`None`)
                pre_transform (callable, optional): A function/transform that takes in
                    an :obj:`torch_geometric.data.Data` object and returns a
                    transformed version. The data object will be transformed before
                    being saved to disk. (default: :obj:`None`)
            """

            def __init__(
                self,
                root: str,
                name: str,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
            ) -> None:
                self.name = name
                assert self.name in ['cal','SF','NA','TG','OL']
                self.edge_url = ('https://users.cs.utah.edu/~lifeifei/research/tpq/{}.cedge')
                self.node_url = ('https://users.cs.utah.edu/~lifeifei/research/tpq/{}.cnode')
                super().__init__(root, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])


            @property
            def raw_dir(self) -> str:
                return osp.join(self.root, self.name, 'raw')

            @property
            def processed_dir(self) -> str:
                return osp.join(self.root, self.name, 'processed')

            @property
            def raw_file_names(self) -> List[str]:
                return [f'{self.name}.cnode', f'{self.name}.cedge']

            @property
            def processed_file_names(self) -> str:
                return 'data.pt'

            def download(self) -> None:
                download_url(self.edge_url.format(self.name), self.raw_dir)
                download_url(self.node_url.format(self.name), self.raw_dir)

            def process(self) -> None:
                index_map, xs = {}, []
                with open(self.raw_paths[0], 'r') as f:
                    rows = f.read().split('\n')[:-1] #see
                    for i, row in enumerate(rows):
                        idx, long, lat = row.split()
                        index_map[int(idx)] = i
                        xs.append([float(long), float(lat)])
                x = torch.tensor(xs)
        #         y = torch.tensor(ys, dtype=torch.long)

                edge_indices, edge_weights = [], []
                with open(self.raw_paths[1], 'r') as f:
                    rows = f.read().split('\n')[:-1] #see
                    for row in rows:
                        _, src, dst, w = row.split()
                        edge_indices.append([index_map[int(src)], index_map[int(dst)]])
                        edge_weights.append(float(w))

                edge_index = torch.tensor(edge_indices).t().contiguous()
                edge_index = coalesce(edge_index, num_nodes=x.size(0))

                edge_weights = torch.tensor(edge_weights)
                edge_weights = 1.01-torch.exp((edge_weights - max(edge_weights))/max(edge_weights))

                data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights)#, y=y
                data = data if self.pre_transform is None else self.pre_transform(data)
                torch.save(self.collate([data]), self.processed_paths[0])

            def __repr__(self) -> str:
                return f'{self.name.capitalize()}Roads()'
        dataset = Roads(root='Pytorch_Tes', name=name.split('_')[1])
        G = dataset[0]
        X_ambient, cluster_labels = G.x, G.y
        X_manifold = X_ambient
    else:
        raise ValueError("Data unknown!!")
    
    return(X_ambient, X_manifold, cluster_labels, G)