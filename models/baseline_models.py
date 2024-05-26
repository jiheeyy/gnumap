from numbers import Number
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
import numpy as np
from abc import ABC
from models.aggregation import *
from models.aggregation import GAPPNP

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers,dropout_rate, normalized= True, gnn_type = "symmetric", alpha = 0.5, beta = 1.0, norm_layer=False):
        super().__init__()
        self.n_layers = n_layers
        self.p = dropout_rate
        self.normalized = normalized
        self.norm_layer, self.batch_norms, self.norm_type = norm_layer, nn.ModuleList(), GraphNorm #GraphNorm # nn.BatchNorm1d
        self.convs = nn.ModuleList()

        if n_layers > 1:
            self.convs.append(GCNConv(in_dim, hid_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
            if norm_layer:
                self.batch_norms.append(self.norm_type(hid_dim))
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
                if norm_layer:
                    self.batch_norms.append(self.norm_type(hid_dim))
            self.convs.append(GCNConv(hid_dim, out_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
            if norm_layer:
                self.batch_norms.append(self.norm_type(out_dim))
        else:
            self.convs.append(GCNConv(in_dim, out_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
            if norm_layer:
                self.batch_norms.append(self.norm_type(out_dim))


    def forward(self, x, edge_index, edge_weight = None):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight)) # nn.PReLU
            if self.norm_layer:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p = self.p)
            if self.normalized is True:
                x = F.normalize(x)
            
        x = self.convs[-1].float()(x.float(), edge_index, edge_weight)
        return x


class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int,
                 activation: str='relu', slope: float=.1,
                 device: str='cpu',
                 alpha_res: float=0, alpha: float=0.5,
                 beta: float=1., gnn_type: str = 'symmetric',
                 norm: str='normalize',
                 separate_neighbors: bool=True,
                 must_propagate: bool=None,
                 lambd_corr: float = 0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.n_layers = n_layers
        self.device = device
        self.alpha_res = alpha_res
        self.alpha = alpha
        self.beta= beta
        self.must_propagate = must_propagate
        self.separate_neighbors = separate_neighbors
        self.propagate = GAPPNP(K=1, alpha_res=self.alpha_res,
                                alpha = self.alpha,
                                gnn_type=self.gnn_type,
                                beta = self.beta)
        self.norm = norm
        if self.must_propagate is None:
            self.must_propagate = [True] * self.n_layers
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
            _fc_list_n = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            _fc_list_n = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                _fc_list_n.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            _fc_list_n.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.fc_n = nn.ModuleList(_fc_list_n)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x, edge_index):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
                if self.norm == 'normalize' and c==0:
                    h = F.normalize(h, p=2, dim=1)
                elif self.norm == 'standardize'and c==0:
                    h = (h - h.mean(0)) / h.std(0)
                elif self.norm == 'uniform'and c==0:
                    h = 10 * (h - h.min()) / (h.max() - h.min())
                elif self.norm == 'col_uniform'and c==0:
                    h = 10 * (h - h.min(0)[0].reshape([1,-1]))/ (h.max(0)[0].reshape([1,-1])-h.min(0)[0].reshape([1,-1]))

            else:
                ### Concatenate representation and new one
                h = self.fc[c](h)
                h = F.dropout(h, p=0.5, training=self.training)
                if self.separate_neighbors:
                    print(c)
                    print(h.shape)
                    print(self.fc_n[0])
                    neighbors = self.fc_n[c](h)
                    neighbors = F.dropout(neighbors, p=0.5, training=self.training)
                if self.must_propagate[c] and self.separate_neighbors == False:
                    h = self.propagate(h, edge_index)
                elif self.separate_neighbors:
                    neighbors = self.propagate(neighbors, edge_index)
                    h += neighbors
                if self.norm == 'normalize':
                    h = F.normalize(h, p=2, dim=1)
                elif self.norm == 'standardize':
                    h = (h - h.mean(0)) / h.std(0) #z1 = (h1 - h1.mean(0)) / h1.std(0)
                elif self.norm == 'uniform':
                    h = 10 * (h - h.min()) / (h.max() - h.min())
                elif self.norm == 'col_uniform':
                    h = 10 * (h - h.min(0)[0].reshape([1,-1]))/ (h.max(0)[0].reshape([1,-1])-h.min(0)[0].reshape([1,-1]))
                h = self._act_f[c](h)
                
        if self.norm == 'standardize_last':
            h = (h - h.mean(0)) / h.std(0)
        return h


class genMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2,
    activation='relu', slope=.1, device='cpu', use_bn=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: torch.nn.ReLU()(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
            _fc_list_n = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            _fc_list_n = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                _fc_list_n.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            _fc_list_n.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.fc_n = nn.ModuleList(_fc_list_n)
        self.to(self.device)
    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
                if self.use_bn: h= self.bn(h)
        return h

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x,_):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

"""Implementation of https://github.com/vlivashkin/pygkernels"""
class Scaler(ABC):
    def __init__(self, A: np.ndarray = None):
        self.eps = 10**-10
        self.A = A

    def scale_list(self, ts):
        for t in ts:
            yield self.scale(t)

    def scale(self, t):
        return t


class Linear(Scaler):  # no transformation, for SP-CT
    pass

class Fraction(Scaler):  # Forest, logForest, Comm, logComm, Heat, logHeat, SCT, SCCT, ...
    def scale(self, t):
        return 0.5 * t / (1.0 - t + self.eps)

def get_D(A):
    """
    Degree matrix
    """
    return np.diag(np.sum(A, axis=0))


def get_L(A):
    """
    Ordinary (or combinatorial) Laplacian matrix.
    L = D - A
    """
    return get_D(A) - A

class Kernel(ABC):
    EPS = 10**-10
    name, _default_scaler = None, None
    _parent_distance_class, _parent_kernel_class = None, None

    def __init__(self, A: np.ndarray):
        assert not (self._parent_distance_class and self._parent_kernel_class)
        if self._parent_distance_class:
            self._parent_kernel = None
            self._parent_distance = self._parent_distance_class(A)
            self._default_scaler = self._parent_distance._default_scaler
        elif self._parent_kernel_class:
            self._parent_kernel = self._parent_kernel_class(A)
            self._parent_distance = None
            self._default_scaler = self._parent_kernel._default_scaler
        self.scaler: Scaler = self._default_scaler(A)
        self.A = A

    def get_K(self, param):
        if self._parent_distance:  # use D -> K transform
            D = self._parent_distance.get_D(param)
            return D_to_K(D)
        elif self._parent_kernel:  # use element-wise log transform
            H0 = self._parent_kernel.get_K(param)
            return ewlog(H0)
        else:
            raise NotImplementedError()

class CT_H(Kernel):
    name, _default_scaler = "CT", Linear

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.K_CT = np.linalg.pinv(get_L(self.A))

    def get_K(self, param=None):
        return self.K_CT

class CCT_H(Kernel):
    name, _default_scaler = "CCT", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.K_CCT = self.H_CCT(A)

    def H_CCT(self, A: np.ndarray):
        """
        H = I - E / n
        M = D^{-1/2}(A - dd^T/vol(G))D^{-1/2},
            d is a vector of the diagonal elements of D,
            vol(G) is the volume of the graph (sum of all elements of A)
        K_CCT = HD^{-1/2}M(I - M)^{-1}MD^{-1/2}H
        """
        size = A.shape[0]
        I = np.eye(size)
        d = np.sum(A, axis=0).reshape((-1, 1))
        D05 = np.diag(np.power(d, -0.5)[:, 0])
        H = np.eye(size) - np.ones((size, size)) / size
        volG = np.sum(A)
        M = D05.dot(A - d.dot(d.transpose()) / volG).dot(D05)
        return H.dot(D05).dot(M).dot(np.linalg.pinv(I - M)).dot(M).dot(D05).dot(H)

    def get_K(self, alpha=None):
        return self.K_CCT
