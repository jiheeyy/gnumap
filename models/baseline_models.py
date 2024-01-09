from numbers import Number
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
    def __init__(self, in_dim, hid_dim, out_dim, n_layers,dropout_rate, normalized= True, gnn_type = "symmetric", alpha = 0.5, beta = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.p = dropout_rate
        self.normalized = normalized
        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(GCNConv(in_dim, hid_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
            print('baseline/firstGCNCONV')
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
                print(print('baseline/middleGCNCONV'))
            self.convs.append(GCNConv(hid_dim, out_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))
            print('baseline/finalGCNCONV')
        else:
            self.convs.append(GCNConv(in_dim, out_dim, gnn_type = gnn_type, alpha = alpha, beta = beta))

    def forward(self, x, edge_index, edge_weight = None):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight)) # nn.PReLU
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
