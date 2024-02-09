import sys,os
sys.setrecursionlimit(3000)
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from frechetdist import frdist
import numba
import scipy.stats
from scipy.spatial.kdtree import distance_matrix
import torch
import seaborn as sns
import pandas as pd
import torch
import torch_geometric
from functools import wraps
from typing import Optional, Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse.csgraph import dijkstra

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from collections import Counter
from numpy.random import default_rng

from scipy.spatial.distance import pdist
from torch_geometric.utils import to_dense_adj
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from scipy import linalg
import time

from gnumap.umap_functions import prob_high_dim, find_ab_params

## Modify original deg function so that they can have >1 dimensional input
def deg(index, num_nodes: Optional[int] = None,           
        dtype: Optional[torch.dtype] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index tensor.
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

def random_triplet_eval(X, X_new, num_triplets=5):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many randomly
    selected triplets have been violated. Each point will generate 5 triplets.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    
    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng()
    triplets = rng.choice(anchors, (X.shape[0], num_triplets, 2))
    triplet_labels = np.zeros((X.shape[0], num_triplets))
    anchors = anchors.reshape((-1, 1, 1))
    
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]
    
    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]

    # Compare the labels and return the accuracy
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/num_triplets
    return acc


def neighbor_kept_ratio_eval(G, X_new, n_neighbors=30):
    '''
    This is a function that evaluates the local structure preservation.
    A nearest neighbor set is constructed on both the high dimensional space and
    the low dimensional space.
    Input:
        G: Origianl graph object
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.

    '''
    if torch.is_tensor(X_new) is True:
        X_new = X_new.detach().numpy()
    nn_ld = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn_ld.fit(X_new)
    # Construct a k-neighbors graph, where 1 indicates a neighbor relationship
    # and 0 means otherwise, resulting in a graph of the shape n * n
    if G.__class__ == torch_geometric.data.data.Data:
        graph_hd = to_dense_adj(G.edge_index, max_num_nodes=G.x.shape[0]).detach() # no self-loops in original graph
    else:
        nn_hd = NearestNeighbors(n_neighbors=n_neighbors+1)
        nn_hd.fit(G)
        grph_hd = nn_hd.kneighbors_graph(G).toarray()
        graph_hd -= np.eye(G.num_nodes) # Removing diagonal
        graph_hd = torch.tensor(graph_hd, dtype= torch.float32)  
    graph_ld = nn_ld.kneighbors_graph(X_new).toarray()
    graph_ld -= np.eye(G.num_nodes) # Removing diagonal
    graph_ld = torch.tensor(graph_ld, dtype= torch.float32)
    neighbor_kept = torch.sum((graph_hd * graph_ld)[0], dim = 0)
    deg_val = deg(G.edge_index, G.num_nodes)
    deg_val[deg_val == 0] = 1e-1
    neighbor_kept_ratio = torch.div(neighbor_kept, deg_val).sum()/G.num_nodes
    return neighbor_kept_ratio


def neighbor_kept_ratio_eval_large(X, X_new, n_neighbors=30, sample_size=10000, seed=0):
    '''
    This is a function that evaluates the local structure preservation.
    In a large dataset, keeping a neighbor graph is infeasible as it will lead
    to OOM error. Therefore, we evaluate the neighborhood using a small portion 
    of points as samples.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        n_neighbors: Number of neighbors considered by the algorithm
        samples: Number of samples considered by the algorithm. 
        seed: The random seed used by the random number generator.
    Output:
        acc: The score generated by the algorithm.
    '''
    rng = np.random.default_rng(seed=seed)
    sample_size = min(X.shape[0], sample_size) # prevent overflow
    indices = rng.choice(np.arange(X.shape[0]), size=sample_size, replace=False)
    correct_cnt = 0 # Counter for intersection
    for i in indices:
        # Calculate the neighbors
        index_list_high = calculate_neighbors(X, i, n_neighbors)
        index_list_low = calculate_neighbors(X_new, i, n_neighbors)

        # Calculate the intersection
        correct_cnt += intersection(index_list_high, index_list_low)
    correct_cnt -= sample_size # Remove self
    neighbor_kept_ratio = correct_cnt / n_neighbors / sample_size
    return neighbor_kept_ratio


def calculate_neighbors(X, i, n_neighbors):
    '''A helper function that calculates the neighbor of a sample in a dataset.
    '''
    if isinstance(i, int):
        diff_mat = X - X[i]
    else:
        diff_mat = X - i # In this case, i is an instance of sample
    # print(f"Shape of the diff matrix is {diff_mat.shape}")
    diff_mat = np.linalg.norm(diff_mat, axis=1)
    diff_mat = diff_mat.reshape(-1)
    # Find the top n_neighbors + 1 entries
    index_list = np.argpartition(diff_mat, n_neighbors + 1)[:n_neighbors+2]
    return index_list


def intersection(index_list1, index_list2):
    '''A helper function that calculates the intersection between two different
    list of indices, with O(n) complexity.'''
    index_dict = {}
    for i in range(len(index_list1)):
        index_dict[index_list1[i]] = 1
    cnt = 0
    for i in range(len(index_list2)):
        if index_list2[i] in index_dict:
            cnt += 1
    return cnt


def neighbor_kept_ratio_series_eval(X, X_news, n_neighbors=30):
    graph_hd = to_dense_adj(X.edge_index).detach()
    nk_ratios = []
    for X_new in X_news:
        nn_ld = NearestNeighbors(n_neighbors=n_neighbors+1)
        nn_ld.fit(X_new)
        graph_ld = nn_ld.kneighbors_graph(X_new).toarray()
        graph_ld -= np.eye(X.shape[0]) # Removing diagonal
        neighbor_kept = np.sum(graph_hd * graph_ld).astype(float)
        neighbor_kept_ratio = neighbor_kept / n_neighbors / X.shape[0]
        nk_ratios.append(neighbor_kept_ratio)
    return nk_ratios


def neighbor_kept_ratio_series_eval_fast(X, X_news, n_neighbors=30):
    nn_hd = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn_hd.fit(X)
    graph_hd = nn_hd.kneighbors(X, return_distance=False)
    graph_hd = graph_hd[:, 1:] # Remove itself
    nk_ratios = []
    for X_new in X_news:
        nn_ld = NearestNeighbors(n_neighbors=n_neighbors+1)
        nn_ld.fit(X_new)
        graph_ld = nn_ld.kneighbors(X_new, return_distance=False)
        graph_ld = graph_ld[:, 1:] # Remove itself
        neighbor_kept = 0
        for i in range(graph_hd.shape[0]):
            neighbor_kept += len(np.intersect1d(graph_hd[i], graph_ld[i]))
        neighbor_kept_ratio = neighbor_kept / n_neighbors / X.shape[0]
        nk_ratios.append(neighbor_kept_ratio)
    return nk_ratios

def knn_eval_plot(G , out, n_neighbors = [3,10,20,50,100]):
    alpha = np.arange(0.1,1.1,0.1)
    res = torch.zeros((len( n_neighbors ),10))
    for j in range(len(n_neighbors)):
        for i in range(10):
            res[j,i] = neighbor_kept_ratio_eval(G, out[i], n_neighbors=n_neighbors[j])
    
    df = pd.DataFrame(res)
    df.columns = np.round(alpha,2)
    df['num_neighbors'] = n_neighbors = [3,10,20,50,100]
    df_unpivot = pd.melt(df, id_vars = 'num_neighbors')
    df_unpivot
    sns.barplot(data = df_unpivot, x = "variable", y = "value", hue = "num_neighbors", palette = "Set2")
    return df
    

def spearman_correlation_eval(G, X_new, random_seed=100): 
    '''Evaluate the global structure of an embedding via spearman correlation in
    distance matrix, following https://www.nature.com/articles/s41467-019-13056-x
    '''
    # Fix the random seed to ensure reproducability
    rng = np.random.default_rng(seed=random_seed)
    dataset_size = G.x.shape[0]

    ind1, ind2 = torch.triu_indices(dataset_size, dataset_size, 1)

    sparse_matrix = to_scipy_sparse_matrix(G.edge_index, num_nodes=G.x.shape[0])
    dist_high = shortest_path(csr_matrix(sparse_matrix),directed = False)[ind1,ind2]
    dist_high[np.isinf(dist_high)] = np.max(dist_high[~np.isinf(dist_high)])*2
    dist_low = distance_matrix(X_new, X_new)[ind1,ind2]
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])
    # Calculate the correlation
    corr, pval = scipy.stats.spearmanr(dist_high, dist_low)
    return dist_high, dist_low, corr, pval


def spearman_correlation_numpy(X, X_new, random_seed=100): 
    '''Evaluate the global structure of an embedding via spearman correlation in
    distance matrix, following https://www.nature.com/articles/s41467-019-13056-x
    '''
    # Fix the random seed to ensure reproducability
    rng = np.random.default_rng(seed=random_seed)
    dataset_size = X_new.shape[0]
    ind1, ind2 = torch.triu_indices(dataset_size, dataset_size, 1)
    dist_high = distance_matrix(X, X)[ind1,ind2]
    dist_low = distance_matrix(X_new, X_new)[ind1,ind2]
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])

    # Calculate the correlation
    corr, pval = scipy.stats.spearmanr(dist_high, dist_low)
    return dist_high, dist_low, corr, pval


def spearman_correlation_series_eval(X, X_news, n_points=1000, random_seed=100):
    corrs = []
    pvals = []
    dist_highs = []
    dist_lows = []    
    for i in range(len(X_news)):
        X_new = X_news[i]
        dist_high, dist_low, corr, pval = spearman_correlation_eval(X, X_new, n_points, random_seed)
        corrs.append(corr)
        pvals.append(pval)
        dist_highs.append(dist_high)
        dist_lows.append(dist_low)
    corrs = np.array(corrs)
    pvals = np.array(pvals)
    dist_highs = np.array(dist_highs)
    dist_lows = np.array(dist_lows)
    return corrs, pvals, dist_highs, dist_lows


def kendall_tau_correlation_eval(X, X_new, n_points=1000, random_seed=100):
    '''Evaluate the global structure of an embedding via spearman correlation in
    distance matrix, following https://www.nature.com/articles/s41467-019-13056-x
    '''
    # Fix the random seed to ensure reproducability
    rng = np.random.default_rng(seed=random_seed)
    dataset_size = X.shape[0]

    # Sample n_points points from the dataset randomly
    sample_index = rng.choice(np.arange(dataset_size), size=n_points, replace=False)

    # Generate the distance matrix in high dim and low dim
    dist_high = distance_matrix(X[sample_index], X[sample_index])
    dist_low = distance_matrix(X_new[sample_index], X_new[sample_index])
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])

    # Calculate the correlation
    corr, pval = scipy.stats.kendalltau(dist_high, dist_low)
    return dist_high, dist_low, corr, pval


def kendall_tau_correlation_series_eval(X, X_news, n_points=1000, random_seed=100):
    corrs = []
    pvals = []
    dist_highs = []
    dist_lows = []    
    for i in range(len(X_news)):
        X_new = X_news[i]
        dist_high, dist_low, corr, pval = kendall_tau_correlation_eval(X, X_new, n_points, random_seed)
        corrs.append(corr)
        pvals.append(pval)
        dist_highs.append(dist_high)
        dist_lows.append(dist_low)
    corrs = np.array(corrs)
    pvals = np.array(pvals)
    dist_highs = np.array(dist_highs)
    dist_lows = np.array(dist_lows)
    return corrs, pvals, dist_highs, dist_lows

def corr_eval_plot(dataset, out,n_points = 1000, random_seed = 1234):
    alpha = np.arange(0.1,1.1,0.1)
    res = torch.zeros((10,3))
    for i in range(10):
        res[i,0] = alpha[i]
        _,_,res[i,1], _ = spearman_correlation_eval(dataset.x, out[i], n_points = n_points, random_seed = random_seed)
        _,_,res[i,2], _ = kendall_tau_correlation_eval(dataset.x, out[i], n_points = n_points, random_seed = random_seed)  
    df = pd.DataFrame(res) 
    df.columns = ['alpha', 'spearman','kendall_tau']
    df_unpivot = pd.melt(df, id_vars = 'alpha', value_vars = ['spearman', 'kendall_tau'])
    sns.barplot(df_unpivot, x = 'alpha', y = 'value', hue = 'variable', palette = 'Set2')
    return df

def centroid_knn_eval(X, X_new, y, k):
    '''Evaluate the global structure of an embedding via the KNC metric:
    neighborhood preservation for cluster centroids, following 
    https://www.nature.com/articles/s41467-019-13056-x
    '''
    # Calculating the cluster centers
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    cluster_mean_ori = np.zeros((num_cat, X.shape[1]))
    cluster_mean_new = np.zeros((num_cat, X_new.shape[1]))
    cnt_ori = np.zeros(num_cat) # number of instances for each class

    # Only loop through the whole dataset once
    for i in range(X.shape[0]):
        ylabel = int(y[i])
        cluster_mean_ori[ylabel] += X[i]
        cluster_mean_new[ylabel] += X_new[i]
        cnt_ori[ylabel] += 1
    cluster_mean_ori = ((cluster_mean_ori.T)/cnt_ori).T
    cluster_mean_new = ((cluster_mean_new.T)/cnt_ori).T

    # Generate the nearest neighbor list in the high dimension
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(cluster_mean_ori)
    _, indices = nbrs.kneighbors(cluster_mean_ori)
    indices = indices[:,1:] # Remove the center itself

    # Now for the low dimension
    nbr_low = NearestNeighbors(n_neighbors=k+1).fit(cluster_mean_new)
    _, indices_low = nbr_low.kneighbors(cluster_mean_new)
    indices_low = indices_low[:,1:] # Remove the center itself

    # Calculate the intersection of both lists
    len_neighbor_list = k * num_cat
    both_nbrs = 0

    # for each category, check each of its indices
    for i in range(num_cat):
        for j in range(k):
            if indices[i, j] in indices_low[i, :]:
                both_nbrs += 1
    # Compare both lists and generate the accuracy
    return both_nbrs/len_neighbor_list


def centroid_knn_series_eval(X, X_news, y, k):
    accs = []
    for i in range(len(X_news)):
        X_new = X_news[i]
        acc = centroid_knn_eval(X, X_new, y, k)
        accs.append(acc)
    accs = np.array(accs)
    return accs


def centroid_corr_eval(X, X_new, y, k):
    '''Evaluate the global structure of an embedding via the KNC metric:
    neighborhood preservation for cluster centroids, following 
    https://www.nature.com/articles/s41467-019-13056-x
    '''
    # Calculating the cluster centers
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    cluster_mean_ori = np.zeros((num_cat, X.shape[1]))
    cluster_mean_new = np.zeros((num_cat, X_new.shape[1]))
    cnt_ori = np.zeros(num_cat) # number of instances for each class

    # Only loop through the whole dataset once
    for i in range(X.shape[0]):
        ylabel = int(y[i])
        cluster_mean_ori[ylabel] += X[i]
        cluster_mean_new[ylabel] += X_new[i]
        cnt_ori[ylabel] += 1
    cluster_mean_ori = ((cluster_mean_ori.T)/cnt_ori).T
    cluster_mean_new = ((cluster_mean_new.T)/cnt_ori).T
    # Generate the distance matrix in high dim and low dim
    dist_high = distance_matrix(cluster_mean_ori, cluster_mean_ori)
    dist_low = distance_matrix(cluster_mean_new, cluster_mean_new)
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])

    # Calculate the correlation
    corr, pval = scipy.stats.spearmanr(dist_high, dist_low)
    return dist_high, dist_low, corr, pval


def eval_reduction_additional_ii(dataset_name, methods):
    print(f'Evaluating {dataset_name}')
    X, y = data_prep(dataset_name)
    if y.shape[0] == 10:
        supervised = False # Not supervised
        num_categories = 0
    else:
        supervised = True
        num_categories = len(np.unique(y))
        k = min((num_categories + 2) // 4, 10) # maximum of 10
    for method in methods:
        # Check if the file exists
        print(method)
        # Skip the evaluation if the method fails to generate a result
        if not os.path.exists(f'./output/{dataset_name}_{method}.npy'):
            print('Result not exist')
            continue
        X_lows = np.load(
            f'./output/{dataset_name}_{method}.npy', allow_pickle=True)

        # Pearson Correlation of the centroids -- supervised
        if supervised:
            print('Centroid Spearman')
            corrs = []
            for i in range(5):
                _, _, corr, _ = centroid_corr_eval(X, X_lows[i], y, k) # k is compatible to the number of categories
                corrs.append(corr)
            corrs = np.array(corrs)
            np.save(f'./results/{dataset_name}_{method}_centroidcorr.npy', corrs)

    print('Finished Successfully')


def eval_reduction_additional(dataset_name, methods):
    print(f'Evaluating {dataset_name}')
    X, y = data_prep(dataset_name)
    if y.shape[0] == 10:
        supervised = False # Not supervised
        num_categories = 0
    else:
        supervised = True
        num_categories = len(np.unique(y))
        k = min((num_categories + 2) // 4, 10) # maximum of 10
    for method in methods:
        # Check if the file exists
        print(method)
        # Skip the evaluation if the method fails to generate a result
        if not os.path.exists(f'./output/{dataset_name}_{method}.npy'):
            print('Result not exist')
            continue
        # Spearman Correlation of the distance matrix -- unsupervised
        print('Spearman Correlation')
        X_lows = np.load(
            f'./output/{dataset_name}_{method}.npy', allow_pickle=True)
        corrs, pvals, dist_highs, dist_lows = spearman_correlation_series_eval(X, X_lows) # default to 1000 points
        corrs = np.array(corrs)
        np.save(f'./results/{dataset_name}_{method}_spearmancorr.npy', corrs)
        # Nearest Neighbor Preservance of the centroids -- supervised
        if supervised:
            print('Centroid KNN')
            accs = centroid_knn_series_eval(X, X_lows, y, k) # k is compatible to the number of categories
            np.save(f'./results/{dataset_name}_{method}_centroidknn.npy', accs)

    print('Finished Successfully')


def eval_reduction_large(dataset_name, methods, metric=0):
    print(f'Evaluating {dataset_name}')
    X, y = data_prep(dataset_name)
    for method in methods:
        # Check if the file exists
        print(method)
        if not os.path.exists(f'./output/{dataset_name}_{method}.npy'):
            continue
        # Unsupervised eval
        # NK Ratio
        X_lows = np.load(
                f'./output/{dataset_name}_{method}.npy', allow_pickle=True)

        if metric == 0:
            print('Nearest Neighbor Kept')
            nk_ratios = []
            for i in range(5):
                nk_ratio = neighbor_kept_ratio_eval_large(X, X_lows[i], sample_size=1000) # 10000 is way too slow
                nk_ratios.append(nk_ratio)
            nk_ratios = np.array(nk_ratios)
            np.save(f'./results/{dataset_name}_{method}_nkratios.npy', nk_ratios)

        # RT Ratio
        elif metric == 1:
            print('Random Triplet Accuracy')
            rte_ratios = []
            for X_low in X_lows:
                rte_ratio = random_triplet_eval(X, X_low)
                rte_ratios.append(rte_ratio)
            rte_ratios = np.array(rte_ratios)
            np.save(f'./results/{dataset_name}_{method}_rteratios.npy', rte_ratios)

        # Supervised eval
        # KNN Acc
        elif metric == 2:
            print('KNN Accuracy')
            knn_accs = []
            for X_low in X_lows:
                knn_acc = knn_eval_large(X_low, y, sample_size=200) # 10-fold, each evaluated with 200 samples
                knn_accs.append(knn_acc)
            np.save(f'./results/{dataset_name}_{method}_knnaccs.npy', knn_accs)

        # SVM Acc
        elif metric == 3:
            print('SVM Accuracy')
            svm_accs = []
            for X_low in X_lows:
                svm_acc = svm_eval_large(X_low, y) # 10-fold evaluated on a subset of 100000 samples
                svm_accs.append(svm_acc)
            np.save(f'./results/{dataset_name}_{method}_svmaccs.npy', svm_accs)
        print('---------')
    print('Finished Successfully')

def prob_low_dim(a,b,Y):
    """
    Compute matrix of probabilities q_ij in low-dimensional space
    """
    inv_distances = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
    return inv_distances

def eval_density_preserve(X, X_news, sigma = 1.0, min_dist = 0.1):
    P = prob_high_dim(sigma = sigma, dist = euclidean_distances(X,X))
    _a, _b = find_ab_params(spread = sigma, min_dist = min_dist)
    Q = prob_low_dim(_a,_b,X_news)
    P = torch.tensor(P, dtype = torch.float)
    Q = torch.tensor(Q, dtype = torch.float)
    x_dist = torch.tensor(euclidean_distances(X,X), dtype = torch.float)
    y_dist = torch.tensor(euclidean_distances(X_news,X_news), dtype = torch.float)
    p_sum = 1/torch.sum(P, dim = 1)
    q_sum = 1/torch.sum(Q, dim = 1)
    R_p = p_sum * torch.sum(torch.mul(P,x_dist),dim = 0)
    R_q = q_sum * torch.sum(torch.mul(Q,y_dist),dim = 0)
    corr,_ = scipy.stats.pearsonr(R_p, R_q)
    return corr

def svm_eval(X, y, n_splits=10, **kwargs):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an SVM with rbf kernel.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the SVM.
    Output:
        acc: The (avg) accuracy generated by an SVM with rbf kernel.
    '''
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        clf = SVC(**kwargs)
        clf.fit(X[train_index], y[train_index])
        acc = clf.score(X[test_index], y[test_index])
        sum_acc += acc
    avg_acc = sum_acc/max_acc
    return avg_acc

def logistic_eval(X, y, n_splits=10, **kwargs):
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    n_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        clf = LogisticRegression(**kwargs)
        clf.fit(X[train_index], y[train_index])
        acc =clf.score(X[test_index], y[test_index]) # R^2 of prediction value 
        sum_acc += acc
    avg_acc = sum_acc/n_acc
    return avg_acc




def regression_eval(X, y, n_splits=10, **kwargs):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an SVM with rbf kernel.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the linear regression.
    Output:
        acc: The (avg) accuracy generated by an linear regression, adjusted R^2
    '''
    X = StandardScaler().fit_transform(X)
    skf = KFold(n_splits=n_splits)
    sum_acc = 0
    n_acc = n_splits
    for train_index, test_index in skf.split(X):
        clf = LinearRegression(**kwargs)
        clf.fit(X[train_index], y[train_index])
        #acc = 1 - ( 1-clf.score(X[test_index], y[test_index]) ) * ( len(y[test_index]) - 1 ) / ( len(y[test_index]) - X.shape[1] - 1 )
        acc =clf.score(X[test_index], y[test_index]) # R^2 of prediction value 
        sum_acc += acc
    avg_acc = sum_acc/n_acc
    return avg_acc

def fretchet_inception_distance(X_manifold, embeds, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    
    Params:
    X_manifold : Ground truth 2D Numpy array
    
    embeds: Predicted embedding 2D Numpy array

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.mean(X_manifold, axis=0)
    sigma1 = np.cov(X_manifold, rowvar=False)
    mu2 = np.mean(embeds, axis=0)
    sigma2 = np.cov(embeds, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def sample_eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,model_name,
             dataset = "Blobs"):
    ### Global metrics
    starting_eval = time.time()
    X_manifold = MinMaxScaler().fit_transform(X_manifold)
    X_ambient = MinMaxScaler().fit_transform(X_ambient)
    embeds = MinMaxScaler().fit_transform(embeds)

    np.random.seed(1)
    sample_indices = np.random.choice(X_manifold.shape[0], 2000, replace=False)
    X_manifold, X_ambient, embeds = X_manifold[sample_indices], X_ambient[sample_indices], embeds[sample_indices]
    try:
        cluster_labels = cluster_labels[sample_indices]
    except:
        cluster_labels = cluster_labels.values.reshape(-1, 1)[sample_indices]
    global_dist = {'acc': svm_eval(embeds, np.array(cluster_labels)),
                    'acc_X': svm_eval(X_ambient,np.array(cluster_labels)),
                    'acc_manifold': svm_eval(X_manifold,np.array(cluster_labels)),
                    'silhouette_embeds': silhouette_score(embeds, np.array(cluster_labels)),
                    'silhouette_X': silhouette_score(X_ambient, np.array(cluster_labels)),
                    'silhouette_manifold': silhouette_score(X_manifold, np.array(cluster_labels)),
                    'calinski_harabasz_score_embeds': calinski_harabasz_score(embeds, np.array(cluster_labels)),
                    'calinski_harabasz_score_X':calinski_harabasz_score(X_ambient, np.array(cluster_labels)), 
                    'calinski_harabasz_score_manifold':calinski_harabasz_score(X_manifold, np.array(cluster_labels)), 
                    'davies_bouldin_score_embeds': davies_bouldin_score(embeds, np.array(cluster_labels)),
                    'davies_bouldin_score_X': davies_bouldin_score(X_ambient, np.array(cluster_labels)),
                    'davies_bouldin_score_manifold': davies_bouldin_score(X_manifold, np.array(cluster_labels)),}
 
    local = {}
    print('Time for sampling part',time.time()-starting_eval)
    return global_dist, local

def eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,model_name,large_class,
             dataset = "Blobs"):
    if large_class:
        print('go to sampling')
        return sample_eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,model_name,dataset)
    else:
        starting_eval = time.time()
        ### Global metrics
        _,_,sp,_ = spearman_correlation_eval(G, embeds)

        X_manifold = MinMaxScaler().fit_transform(X_manifold)
        X_ambient = MinMaxScaler().fit_transform(X_ambient)
        embeds = MinMaxScaler().fit_transform(embeds)
        if embeds.shape[1] == 3 or dataset in [ "Cora", "Pubmed", "Citeseer",'Products',
        "Mouse1","Mouse2","Mouse3","Cancer"]:
            sp_manifold = np.nan
            fr_dist =  np.nan
            curve_dist = np.nan

        elif dataset in ["Trefoil", "Helix", "Swissroll", "Sphere", "Helix",
                    "Blobs", "Moons", "Circles"]:
            _,_, sp_manifold, _ = spearman_correlation_numpy(X_manifold, embeds)
            fr_dist = fretchet_inception_distance(X_manifold, embeds)
            curve_dist = np.square(X_manifold -  embeds).mean()

        global_dist = {'frechet': fr_dist,
                        'distance_between_curves': curve_dist,
                        'acc': svm_eval(embeds, np.array(cluster_labels)),
                        'acc_X': svm_eval(X_ambient,np.array(cluster_labels)),
                        'acc_manifold': svm_eval(X_manifold,np.array(cluster_labels)),
                        'acc_linear': logistic_eval(embeds, np.array(cluster_labels), n_splits=10, penalty=None),
                        'acc_linear_X': logistic_eval(X_ambient, np.array(cluster_labels), n_splits=10, penalty=None),
                        'acc_linear_manifold': logistic_eval(X_manifold, np.array(cluster_labels), n_splits=10, penalty=None),
                        'silhouette_embeds': silhouette_score(embeds, np.array(cluster_labels)),
                        'silhouette_X': silhouette_score(X_ambient, np.array(cluster_labels)),
                        'silhouette_manifold': silhouette_score(X_manifold, np.array(cluster_labels)),
                        'calinski_harabasz_score_embeds': calinski_harabasz_score(embeds, np.array(cluster_labels)),
                        'calinski_harabasz_score_X':calinski_harabasz_score(X_ambient, np.array(cluster_labels)), 
                        'calinski_harabasz_score_manifold':calinski_harabasz_score(X_manifold, np.array(cluster_labels)), 
                        'davies_bouldin_score_embeds': davies_bouldin_score(embeds, np.array(cluster_labels)),
                        'davies_bouldin_score_X': davies_bouldin_score(X_ambient, np.array(cluster_labels)),
                        'davies_bouldin_score_manifold': davies_bouldin_score(X_manifold, np.array(cluster_labels)),
                        'spearman_graph': sp,
                        'spearman_manifold': sp_manifold}

        local = {}
        for i, n_neighbors in enumerate([1, 3, 5, 10, 20, 30, 50]):
            local['neighbor_'  + str(n_neighbors)] = float(neighbor_kept_ratio_eval(G, embeds, 
                                                                                    n_neighbors = n_neighbors).detach().numpy())
        density = eval_density_preserve(X_manifold, embeds)
        ### try another density evaluation metric by cluster
        average_distance_manifold = [None] * len(np.unique(cluster_labels))
        average_distance_embeds = [None] * len(np.unique(cluster_labels))
        for u, c in enumerate(np.unique(cluster_labels)):
            distances = pdist(X_manifold[np.where(cluster_labels == c)[0], :])
            average_distance_manifold[u] = np.mean(distances)
            distances_embeds = pdist(embeds[np.where(cluster_labels == c)[0], :])
            average_distance_embeds[u] = np.mean(distances_embeds)
            
        average_distance_manifold = np.array(average_distance_manifold)[~np.isnan(average_distance_manifold)]
        average_distance_embeds = np.array(average_distance_embeds)[~np.isnan(average_distance_embeds)]
        
        local['average_density_X'] = np.mean(average_distance_manifold[~np.isnan(average_distance_manifold)])
        local['average_density_embeds'] = np.mean(average_distance_embeds)
        local['min_density_X'] = np.min(average_distance_manifold)
        local['min_density_embeds'] = np.min(average_distance_embeds)
        local['max_density_X'] = np.max(average_distance_manifold)
        local['max_density_embeds'] = np.max(average_distance_embeds)
        local['median_density_X'] = np.median(average_distance_manifold)
        local['median_density_embeds'] = np.median(average_distance_embeds)
        local['q25_density_X'] = np.quantile(average_distance_manifold, 0.25)
        local['q25_density_embeds'] = np.quantile(average_distance_embeds, 0.25)
        local['q75_density_X'] = np.quantile(average_distance_manifold, 0.75)
        local['q75_density_embeds'] = np.quantile(average_distance_embeds, 0.75)
        local['corr_density'] = np.corrcoef(average_distance_manifold, average_distance_embeds)[0,1]
        print('Time to original eval_all', time.time()-starting_eval)
        return global_dist, local
