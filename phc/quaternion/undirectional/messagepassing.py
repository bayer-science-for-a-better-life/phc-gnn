import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj, Size
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_sum
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, List, Dict

from phc.quaternion.algebra import QTensor
from phc.quaternion.layers import QMLP, QLinear
from phc.quaternion.activations import get_module_activation

from typing import Optional



class QGNNConv(MessagePassing):
    r"""
    Modified Quaternion graph convolution operator as introduced by Nguyen et al. (2020)
    - `"Quaternion Graph Neural Networks" <https://arxiv.org/abs/2008.05089>` -.
    Uses a aggregation first and then a linear trafo.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 add_self_loops: bool = True, init: str = "orthogonal", aggr: Optional[str] = "add",
                 same_dim: bool = True, msg_encoder="identity") -> None:
        super(QGNNConv, self).__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.init = init
        self.aggr = aggr
        self.transform = QLinear(in_features=in_features, out_features=out_features, bias=bias, init=init)
        self.same_dim = same_dim
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, q: QTensor, edge_index: Adj, edge_attr: QTensor, size: Size = None) -> QTensor:

        assert edge_attr.__class__.__name__ == "QTensor"
        x = q.clone()
        # propagate each part, i.e. real part and the three complex parts
        agg_r = self.propagate(edge_index=edge_index, x=q.r, edge_attr=edge_attr.r,
                               size=size)  # [b_num_nodes, in_features]
        agg_i = self.propagate(edge_index=edge_index, x=q.i, edge_attr=edge_attr.i,
                               size=size)  # [b_num_nodes, in_features]
        agg_j = self.propagate(edge_index=edge_index, x=q.j, edge_attr=edge_attr.j,
                               size=size)  # [b_num_nodes, in_features]
        agg_k = self.propagate(edge_index=edge_index, x=q.k, edge_attr=edge_attr.k,
                               size=size)  # [b_num_nodes, in_features]

        q = QTensor(agg_r, agg_i, agg_j, agg_k)

        if self.same_dim:  # aggregate messages -> linearly transform -> add self-loops.
            q = self.transform(q)
            if self.add_self_loops:
                q += x
        else:
            if self.add_self_loops:  # aggregate messages -> add self-loops -> linearly transform.
                q += x
            q = self.transform(q)

        return q

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def __repr__(self):
        return "{}(in_features={}, out_features={}, bias={}, add_self_loops={}, " \
               "activation='{}', init='{}', aggr='{}, same_dim={}')".format(self.__class__.__name__,
                                                                            self.in_features,
                                                                            self.out_features,
                                                                            self.bias,
                                                                            self.add_self_loops,
                                                                            self.activation_str,
                                                                            self.init,
                                                                            self.aggr, self.same_dim)


class QGINEConv(MessagePassing):
    r"""
    Quaternion graph convolution operator similar to the GIN convolution, where the node-features are first
    aggregated and then transformed using a 2-layer quaternion MLP.
    This implementation differs from the introduced graph quaternion convolution operator described from
    Nguyen et al. (2020) - `"Quaternion Graph Neural Networks" <https://arxiv.org/abs/2008.05089>` -.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, add_self_loops: bool = True,
                 norm: [Optional] = None, activation: str = "relu", init: str = "orthogonal", aggr: str = "add",
                 msg_encoder="identity") -> None:
        super(QGINEConv, self).__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.norm = norm
        self.activation_str = activation
        self.init = init
        self.aggr = aggr
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)
        self.transform = QMLP(in_features=in_features, out_features=out_features, factor=1, bias=bias,
                              activation=activation, norm=norm, init=init)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, q: QTensor, edge_index: Adj, edge_attr: QTensor, size: Size = None) -> QTensor:

        assert edge_attr.__class__.__name__ == "QTensor"

        # propagate each part, i.e. real part and the three complex parts
        agg_r = self.propagate(edge_index=edge_index, x=q.r, edge_attr=edge_attr.r,
                               size=size)  # [b_num_nodes, in_features]
        agg_i = self.propagate(edge_index=edge_index, x=q.i, edge_attr=edge_attr.i,
                               size=size)  # [b_num_nodes, in_features]
        agg_j = self.propagate(edge_index=edge_index, x=q.j, edge_attr=edge_attr.j,
                               size=size)  # [b_num_nodes, in_features]
        agg_k = self.propagate(edge_index=edge_index, x=q.k, edge_attr=edge_attr.k,
                               size=size)  # [b_num_nodes, in_features]
        aggregated = QTensor(agg_r, agg_i, agg_j, agg_k)

        if self.add_self_loops:
            q += aggregated

        # transform aggregated node embeddings
        q = self.transform(q)
        return q


    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def __repr__(self):
        return "{}(in_features={}, out_features={}, bias={}, add_self_loops={}, " \
               "norm='{}', activation='{}', init='{}', aggr='{}')".format(self.__class__.__name__,
                                                                          self.in_features,
                                                                          self.out_features,
                                                                          self.bias,
                                                                          self.add_self_loops,
                                                                          self.norm,
                                                                          self.activation_str,
                                                                          self.init,
                                                                          self.aggr)


class QGNNConvSoftmax(MessagePassing):
    r"""
       Quaternion graph convolution operator that uses the softmax aggregation schema.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 add_self_loops: bool = True, init: str = "orthogonal", aggr: Optional[str] = "softmax",
                 same_dim: bool = True, msg_encoder: str = "identity", **kwargs) -> None:
        super(QGNNConvSoftmax, self).__init__(aggr=None)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.init = init
        self.aggr = aggr
        self.transform = QLinear(in_features=in_features, out_features=out_features, bias=bias, init=init)
        self.same_dim = same_dim
        self.initial_beta = kwargs.get("initial_beta")
        self.learn_beta = kwargs.get("learn_beta")
        self.beta = nn.Parameter(torch.tensor(self.initial_beta), requires_grad=self.learn_beta)
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()
        self.beta.data.fill_(self.initial_beta)


    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
        out = scatter_softmax(inputs * self.beta, index, dim=self.node_dim)
        out = scatter_sum(inputs * out, index, dim=self.node_dim, dim_size=dim_size)
        return out

    def forward(self, q: QTensor, edge_index: Adj, edge_attr: QTensor, size: Size = None) -> QTensor:

        assert edge_attr.__class__.__name__ == "QTensor"
        x = q.clone()
        # "cast" QTensor back to torch.Tensor
        q = q.stack(dim=1)   # (batch_num_nodes, 4, feature_dim)
        q = q.reshape(q.size(0), -1)  # (batch_num_nodes, 4*feature_dim)

        edge_attr = edge_attr.stack(dim=1)
        edge_attr = edge_attr.reshape(edge_attr.size(0), -1)

        # propagate
        agg = self.propagate(edge_index=edge_index, x=q, edge_attr=edge_attr, size=size)
        agg = agg.reshape(agg.size(0), 4, -1).permute(1, 0, 2)

        q = QTensor(*agg)

        if self.same_dim:  # aggregate messages -> linearly transform -> add self-loops.
            q = self.transform(q)
            if self.add_self_loops:
                q += x
        else:
            if self.add_self_loops:  # aggregate messages -> add self-loops -> linearly transform.
                q += x
            q = self.transform(q)

        return q


    def __repr__(self):
        return "{}(in_features={}, out_features={}, bias={}, add_self_loops={}, " \
               "activation='{}', init='{}', aggr='{}, same_dim={}')".format(self.__class__.__name__,
                                                                            self.in_features,
                                                                            self.out_features,
                                                                            self.bias,
                                                                            self.add_self_loops,
                                                                            self.activation_str,
                                                                            self.init,
                                                                            self.aggr, self.same_dim)


class QGINEConvSoftmax(MessagePassing):
    r"""
    Quaternion graph convolution operator that utilizes a softmax aggregator and a MLP network as post-trafo.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, add_self_loops: bool = True,
                 norm: [Optional] = None, activation: str = "relu", init: str = "orthogonal", aggr: str = "softmax",
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(QGINEConvSoftmax, self).__init__(aggr=None)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.norm = norm
        self.activation_str = activation
        self.init = init
        self.aggr = aggr
        self.transform = QMLP(in_features=in_features, out_features=out_features, factor=1, bias=bias,
                              activation=activation, norm=norm, init=init)

        self.initial_beta = kwargs.get("initial_beta")
        self.learn_beta = kwargs.get("learn_beta")
        self.beta = nn.Parameter(torch.tensor(self.initial_beta), requires_grad=self.learn_beta)
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()
        self.beta.data.fill_(self.initial_beta)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
        out = scatter_softmax(inputs * self.beta, index, dim=self.node_dim)
        out = scatter_sum(inputs * out, index, dim=self.node_dim, dim_size=dim_size)
        return out

    def forward(self, q: QTensor, edge_index: Adj, edge_attr: QTensor, size: Size = None) -> QTensor:


        assert edge_attr.__class__.__name__ == "QTensor"
        x = q.clone()
        # "cast" QTensor back to torch.Tensor
        q = q.stack(dim=1)  # (batch_num_nodes, 4, feature_dim)
        q = q.reshape(q.size(0), -1)  # (batch_num_nodes, 4*feature_dim)

        edge_attr = edge_attr.stack(dim=1)
        edge_attr = edge_attr.reshape(edge_attr.size(0), -1)

        # propagate
        agg = self.propagate(edge_index=edge_index, x=q, edge_attr=edge_attr, size=size)
        agg = agg.reshape(agg.size(0), 4, -1).permute(1, 0, 2)

        agg = QTensor(*agg)

        if self.add_self_loops:
            x += agg

        # transform aggregated node embeddings
        q = self.transform(x)
        return q


    def __repr__(self):
        return "{}(in_features={}, out_features={}, bias={}, add_self_loops={}, " \
               "norm='{}', activation='{}', init='{}', aggr='{}')".format(self.__class__.__name__,
                                                                          self.in_features,
                                                                          self.out_features,
                                                                          self.bias,
                                                                          self.add_self_loops,
                                                                          self.norm,
                                                                          self.activation_str,
                                                                          self.init,
                                                                          self.aggr)


class QMessagePassing(nn.Module):
    r""" Wraps the optional quaternion graph convolution operators into one class """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, add_self_loops: bool = True,
                 norm: [Optional] = None, activation: str = "relu", init: str = "orthogonal", aggr: str = "add",
                 mlp: bool = True, same_dim: bool = True, msg_encoder: str = "identity",  **kwargs):
        super(QMessagePassing, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.norm = norm
        self.activation_str = activation
        self.init = init
        self.aggr = aggr
        self.mlp = mlp
        self.same_dim = same_dim
        self.msg_encoder = msg_encoder

        if aggr == "softmax":
            if mlp:
                self.transform = QGINEConvSoftmax(in_features, out_features, bias, add_self_loops, norm, activation,
                                                  init, aggr, msg_encoder, **kwargs)
            else:
                self.transform = QGNNConvSoftmax(in_features, out_features, bias, add_self_loops, init, aggr,
                                                 same_dim, msg_encoder,  **kwargs)
        else:
            if mlp:
                self.transform = QGINEConv(in_features, out_features, bias, add_self_loops, norm, activation, init,
                                           aggr, msg_encoder)
            else:
                self.transform = QGNNConv(in_features, out_features, bias, add_self_loops, init, aggr, same_dim,
                                          msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, q: QTensor, edge_index: Adj, edge_attr: QTensor, size: Size = None) -> QTensor:
        return self.transform(q, edge_index, edge_attr, size)
