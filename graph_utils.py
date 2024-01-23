import numpy as np
import torch
import copy
from torch_scatter import scatter_add
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric.transforms as T
from gnumap.umap_functions import *
from typing import Optional, Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, to_networkx
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn import manifold


def get_weights(data, neighbours=15, method = 'laplacian', beta=1,
                alpha=0.5, power = 3):
    if method == 'heat':
        data.edge_weight = torch.ones(data.edge_index.shape[1])
        transform = T.GDC(
                self_loop_weight=beta,
                normalization_in='row',
                normalization_out='row',
                diffusion_kwargs=dict(method='ppr', alpha=0.2), # maybe add alpha as parameter?
                sparsification_kwargs=dict(method='topk', k= neighbours, dim=0), # k as parameter
                exact=True,
            )
        newA = transform(copy.deepcopy(data))
        edge_index, edge_weights = transform_edge_weights(newA.edge_index,
                                                          newA.edge_attr,
                                                          newA.num_nodes,
                                                          n_neighbours = neighbours)

    elif method == 'power':
        A = to_scipy_sparse_matrix(data.edge_index)
        A2 = A.dot(A)
      #  A3 = A.dot(A2)
        A3 = A**power
        new_edge_index, new_edge_weights =  from_scipy_sparse_matrix(A3)
        new_edge_weights = torch.exp(-(new_edge_weights-new_edge_weights.max())/
             torch.median(new_edge_weights-new_edge_weights.max()))
        edge_index, edge_weights = transform_edge_weights(new_edge_index,
                                                          new_edge_weights,
                                                          data.num_nodes,
                                                          n_neighbours = neighbours)
        ####  This is perhaps not the best way to create my network though...,
        #### Maybe not the best weights,

    elif method == 'laplacian':
        # num_nodes = data.num_nodes,
        edge_index0 = data.edge_index
        edge_weight = torch.ones(edge_index0.shape[1])
        edge_index, edge_weight = add_remaining_self_loops(
                         edge_index0, edge_weight, beta, data.num_nodes)
        row, col = edge_index[0], edge_index[1]
        #edge_weights = torch.ones(row.shape[0]),
        deg = scatter_add(edge_weight, col, dim=0, dim_size=data.num_nodes)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        L = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        #L = deg_inv_sqrt[row] * edge_weight #* deg_inv_sqrt[col],

        P = to_scipy_sparse_matrix(edge_index, edge_attr=L, num_nodes=data.num_nodes)
        A = P + P.T - P.multiply(P.T)
        edge_index, edge_weights  = from_scipy_sparse_matrix(A)#,
        edge_index, edge_weights = transform_edge_weights(edge_index,
                                                          edge_weights,
                                                          data.num_nodes,
                                                          n_neighbours = neighbours)
    else:
        raise ValueError("Model unknown!!")

    edge_index, edge_weights = add_remaining_self_loops(
                  edge_index,
                  edge_weights,
                  beta, data.num_nodes)

    return(edge_index, edge_weights)


def transform_edge_weights(edge_index, edge_weight, num_nodes, n_neighbours = 15):
    rows = []
    cols = []
    weights = []
    sigmas = []
    n_eff_neighbours = []
    for u in range(num_nodes):
        rows+= list(edge_index[0, edge_index[0,:] == u].numpy())
        cols+= list(edge_index[1, edge_index[0,:] == u].numpy())
        #### want to find the appropriate scaling factor,
        dist_row = edge_weight[edge_index[0,:] == u].numpy()
        func = lambda sigma: k(prob_high_dim(sigma, dist_row))
        binary_search_result = sigma_binary_search(func, n_neighbours) #### Maybe we should have a varying number of neighbours here,
        sigmas += [binary_search_result]
        weights += list(prob_high_dim(binary_search_result, dist_row))
        n_eff_neighbours += [k(prob_high_dim(binary_search_result, dist_row))]

    edge_weights = torch.from_numpy(np.array(weights))
    edge_index = torch.vstack([torch.from_numpy(np.array(rows)),
                          torch.from_numpy(np.array(cols))]).long()
    return(edge_index, edge_weights)


def deg(index, num_nodes: Optional[int] = None,           
        dtype: Optional[torch.dtype] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index tensor.
    ## Modify original deg function so that they can have >1 dimensional input
    Args:
    index (LongTensor): Index tensor.
    num_nodes (int, optional): The number of nodes, *i.e.*
    :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dtype (:obj:`torch.dtype`, optional): The desired data type of the
    returned tensor.\n\n    :rtype: :class:`Tensor`\n    """
    if index.shape[0] != 1: # modify input 
        index = index[0] 
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def convert_to_graph(X, n_neighbours =15, features='none', standardize=True,
                     radius_knn = 0., featdim = 0, bw = None):                
    n = X.shape[0]
    if standardize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
    if radius_knn > 0 :
        A = radius_neighbors_graph(X, radius = radius_knn, mode='distance', include_self=False) # edge weight is given by a transformation of the distance
    else:
        A = kneighbors_graph(X, n_neighbours, mode='distance', include_self=False) # edge weight is given by distance
    
    edge_index, edge_weights = from_scipy_sparse_matrix(A)
    edge_index, edge_weights = to_undirected(edge_index, edge_weights)
    if bw is None:
        bw = torch.max(edge_weights)
    if features == 'coordinates':
        feats = torch.from_numpy(X).float()
    elif features == 'ones':
        feats = torch.ones(n, featdim)
    elif features== 'lap':
        lap = manifold.SpectralEmbedding(n_components=10, n_neighbors=n_neighbours)
        feats = lap.fit_transform(X)
        feats = torch.tensor(feats).float()
    else:
        feats = torch.eye(n)
        
    new_data = Data(x=feats, edge_index=edge_index, # 
                    edge_weight= torch.exp((edge_weights - max(edge_weights))/max(edge_weights)), #torch.exp(-(edge_weights**2)/(2 * bw**2)),
                    sparse=A.toarray()) # heat kernel
    ## edge_weight kernel transf 0 1 todo
    return new_data

