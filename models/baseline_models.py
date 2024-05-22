from numbers import Number
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from models.aggregation import *
from models.aggregation import GAPPNP

import numpy as np
from sklearn.utils import deprecated
from abc import ABC
from scipy.linalg import expm

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


class AlphaToT(Scaler):  # α > 0 -> 0 < t < α^{-1}
    def __init__(self, A: np.ndarray = None):
        super().__init__(A)
        cfm = np.linalg.eigvals(self.A)
        self.rho = np.max(np.abs(cfm))

    def scale(self, alpha):
        return 1 / ((1 / alpha + self.rho + self.eps) + self.eps)


class Rho(Scaler):  # pWalk, Walk
    def __init__(self, A: np.ndarray = None):
        super().__init__(A)
        cfm = np.linalg.eigvals(self.A)
        self.rho = np.max(np.abs(cfm))

    def scale(self, t):
        return t / (self.rho + self.eps)


class Fraction(Scaler):  # Forest, logForest, Comm, logComm, Heat, logHeat, SCT, SCCT, ...
    def scale(self, t):
        return 0.5 * t / (1.0 - t + self.eps)


class FractionReversed(Scaler):  # RSP, FE
    def scale(self, beta):
        return (1.0 - beta) / (beta + self.eps)

@deprecated()
def normalize(dm):
    return dm / dm.std() if dm.std() != 0 else dm


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


def get_normalized_L(A):
    """
    Normalized Laplacian matrix.
    L = D^{-1/2}*L*D^{-1/2}
    """
    D = get_D(A)
    L = get_L(A)
    D_12 = np.linalg.inv(np.sqrt(D))
    return D_12.dot(L).dot(D_12)


def get_P(A):
    """
    Markov matrix.
    P = D^{-1}*A
    """
    D = get_D(A)
    return np.linalg.inv(D).dot(A)


def ewlog(K):
    """
    logK = element-wise log(K)
    """
    mask = K <= 0
    K[mask] = 1
    logK = np.log(K)
    logK[mask] = -np.inf
    return logK


def K_to_D(K):
    """
    D = (k * 1^T + 1 * k^T - K - K^T) / 2
    k = diag(K)
    """
    size = K.shape[0]
    k = np.diagonal(K).reshape(-1, 1)
    i = np.ones((size, 1))
    return 0.5 * ((k.dot(i.transpose()) + i.dot(k.transpose())) - K - K.transpose())


def D_to_K(D):
    """
    K = -1/2 H*D*H
    H = I - E/n
    """
    size = D.shape[0]
    I, E = np.eye(size), np.ones((size, size))
    H = I - (E / size)
    K = -0.5 * H.dot(D).dot(H)
    return K

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


class Katz_H(Kernel):
    name, _default_scaler = "Katz", Rho

    def get_K(self, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * self.A)


class For_H(Kernel):
    name, _default_scaler = "For", Fraction

    def get_K(self, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.inv(np.eye(size) + t * get_L(self.A))


class Comm_H(Kernel):
    name, _default_scaler = "Comm", Fraction

    def get_K(self, t):
        """
        H0 = exp(tA)
        """
        return expm(t * self.A)  # if t < 30 else None


class Heat_H(Kernel):
    name, _default_scaler = "Heat", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.L = get_L(self.A)

    def get_K(self, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * self.L)


class NHeat_H(Kernel):
    name, _default_scaler = "NHeat", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.nL = get_normalized_L(A)

    def get_K(self, t):
        """
        H0 = exp(-t*nL)
        """
        return expm(-t * self.nL)


class SCT_H(CT_H):
    name, _default_scaler = "SCT", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.sigma = self.K_CT.std()
        self.Kds = self.K_CT / (self.sigma + self.EPS)

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1.0 / (1.0 + np.exp(-alpha * self.Kds))


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
        return dot(D05).dot(M).dot(np.linalg.pinv(I - M)).dot(M).dot(D05).dot(H)

    def get_K(self, alpha=None):
        return self.K_CCT


class SCCT_H(CCT_H):
    name, _default_scaler = "SCCT", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.sigma = self.K_CCT.std()
        self.Kds = self.K_CCT / self.sigma

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1.0 / (1.0 + np.exp(-alpha * self.Kds))


class PPR_H(Kernel):
    name, _default_scaler = "PPR", Linear

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = get_P(A)

    def get_K(self, alpha):
        """
        H = (I - αP)^{-1}
        """
        return np.linalg.inv(self.I - alpha * self.P)


class ModifPPR_H(Kernel):
    name, _default_scaler = "ModifPPR", Linear

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.D = get_D(A)

    def get_K(self, alpha):
        """
        H = (I - αP)^{-1}*D^{-1} = (D - αA)^{-1}
        """
        return np.linalg.inv(self.D - alpha * self.A)


class HeatPR_H(Kernel):
    name, _default_scaler = "HeatPR", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = get_P(A)

    def get_K(self, t):
        """
        H = expm(-t(I - P))
        """
        return expm(-t * (self.I - self.P))


class DF_H(Kernel):
    name, _default_scaler = "DF", Fraction

    def __init__(self, A: np.ndarray, n_iter=30):
        super().__init__(A)
        self.n_iter = n_iter
        self.dfac = self.calc_double_factorial(n_iter)

    @staticmethod
    def calc_double_factorial(max_k):
        mem = np.zeros((max_k + 1,))
        mem[0], mem[1] = 1, 1
        for i in range(2, max_k + 1):
            mem[i] = mem[i - 2] * i
        return mem

    def get_K(self, t):
        tA = t * self.A
        K, tA_k = np.eye(tA.shape[0]), np.eye(tA.shape[0])
        for i in range(1, self.n_iter):
            tA_k = tA_k.dot(tA)
            K += tA_k / self.dfac[i]
        return K


class Abs_H(Kernel):
    name, _default_scaler = "Abs", Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.L = get_L(A)

    def get_K(self, t):
        return np.linalg.pinv(t * self.A + self.L)
