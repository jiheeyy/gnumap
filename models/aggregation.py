from typing import Optional, Tuple

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes



@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, alpha=0.5, beta=1.,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = beta

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(fill_value, dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def gcn_norm_rw(edge_index, edge_weight=None, num_nodes=None, alpha=0.5, beta=1.,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = beta

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.) #put 0. where deg_inv_sprt == inf
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))  #### L_alpha = D^{-alpha}LD^{-alpha}
        deg = sparsesum(adj_t, dim=1)  #### degree D_alpha
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        L=  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        deg_alpha = scatter_add(L, row, dim=0, dim_size= num_nodes)
        deg_inv = deg_alpha.pow_(-1)
        deg_inv.masked_fill_(deg == float('inf'), 0)
        L =  deg_inv[row] * L
        return edge_index , L


def gcn_diffusion(edge_index,edge_weight=None, num_nodes=None,
              alpha=0.5, beta=1.0,
             add_self_loops=True, dtype=None):

    fill_value = beta

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1) # sum of each row of the sparse tensor
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.) #put 0. where deg_inv_sprt == inf
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))  #### L_alpha = D^{-alpha}LD^{-alpha}
        deg = sparsesum(adj_t, dim=1)  #### degree D_alpha
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv.view(1, -1))
        if add_self_loops == False:
            adj_t = fill_diag(adj_t, 1.)
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size= num_nodes)
        #print(deg)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        L=  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        #print(scatter_add(L, col, dim=0, dim_size=data.num_nodes))
        deg_alpha = scatter_add(L, row, dim=0, dim_size= num_nodes)
        deg_inv = deg_alpha.pow_(-1)
        deg_inv.masked_fill_(deg == float('inf'), 0)
        L =  deg_inv[row] * L
        if add_self_loops == False:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, L, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            L = tmp_edge_weight
        return edge_index, L


def gcn_norm_sym(edge_index, edge_weight=None, num_nodes=None,  alpha=0.5, beta=1.0,
             add_self_loops=True, dtype=None):

    fill_value = beta

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1) # sum of each row of the sparse tensor
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.) #put 0. where deg_inv_sprt == inf
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))  #### L_alpha = D^{-alpha}LD^{-alpha}
        deg = sparsesum(adj_t, dim=1)  #### degree D_alpha
        deg_inv = deg.pow_(-0.5)
        deg_inv.masked_fill_(deg == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv.view(1, -1))
        adj_t = mul(adj_t, deg_inv.view(-1, 1))
        if add_self_loops == False:
            adj_t = fill_diag(adj_t, 1.)
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size= num_nodes)
        #print(deg)
        deg_inv_sqrt = deg.pow_(-alpha)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        L=  deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        #print(scatter_add(L, col, dim=0, dim_size=data.num_nodes))
        deg_alpha = scatter_add(L, row, dim=0, dim_size= num_nodes)
        deg_inv = deg_alpha.pow_(-0.5)
        deg_inv.masked_fill_(deg == float('inf'), 0)
        L =  deg_inv[row] * L * deg_inv[col]
        if add_self_loops == False:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, L, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            L = tmp_edge_weight
        return edge_index, L




class GAPPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha_res: float, alpha: float=0.5, beta: float = 1.0, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 gnn_type: str = "symmetric",
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha_res = alpha_res
        self.alpha = alpha
        self.beta= beta
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.gnn_type = gnn_type
        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    if self.gnn_type == "symmetric":
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), self.alpha, self.beta,
                            self.add_self_loops, self.flow, dtype=x.dtype)
                    elif self.gnn_type == "RW":
                     #   print("here")
                        edge_index, edge_weight = gcn_norm_rw(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.alpha, self.beta, self.add_self_loops)
                    elif self.gnn_type == "symmetrized RW":
                     #   print("here")
                        edge_index, edge_weight = gcn_norm_sym(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            self.alpha, beta=self.beta, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    if self.gnn_type == "symmetric":
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            alpha = self.alpha, beta=self.beta,
                            add_self_loops=self.add_self_loops,
                            flow=self.flow, dtype=x.dtype)
                    elif self.gnn_type == "RW":
                      #  print("here")
                        edge_index, edge_weight = gcn_norm_rw(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            alpha = self.alpha, beta=self.beta, add_self_loops=self.add_self_loops)
                    elif self.gnn_type == "symmetrized RW":
                      #  print("here")
                        edge_index, edge_weight = gcn_norm_sym(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim),
                            alpha = self.alpha, beta=self.beta, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha_res)
            x += self.alpha_res * h

        return x


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha_res={self.alpha_res}, alpha={self.alpha}, beta={self.beta})'



class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 gnn_type: str = "symmetric", alpha: float = 0.5, beta: float = 1.0,
                 bias: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = beta
        self.cached = cached
        self.gnn_type =  gnn_type
        self.alpha = alpha
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        print('aggregation/in_channels',in_channels)
        print('aggregation/out_channels',out_channels)
        self.lin = Linear(in_channels, out_channels, bias=False)
        print('lin')
        torch.nn.init.xavier_normal_(self.lin.weight, gain=0.003)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    if self.gnn_type == "symmetric":
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), self.alpha, self.beta,
                            self.add_self_loops, self.flow, dtype=x.dtype)
                    elif self.gnn_type == "RW":
                       # print("here")
                        edge_index, edge_weight = gcn_norm_rw(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            self.alpha, self.beta, self.add_self_loops)
                    elif self.gnn_type == "symmetrized RW":
                       # print("here")
                        edge_index, edge_weight = gcn_norm_sym(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            alpha = self.alpha, beta=self.beta, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    if self.gnn_type == "symmetric":
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), 
                            self.alpha, self.beta, self.add_self_loops, self.flow, dtype=x.dtype)
                    elif self.gnn_type == "RW":
                      #  print("here")
                        edge_index, edge_weight = gcn_norm_rw(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            self.alpha, self.beta, self.add_self_loops)
                    elif self.gnn_type == "symmetrized RW":
                        # print("here")
                        edge_index, edge_weight = gcn_norm_sym(  # yapf: disable
                            edge_index,  edge_weight, x.size(self.node_dim),
                            alpha =  self.alpha, beta=self.beta, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index = cache

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

