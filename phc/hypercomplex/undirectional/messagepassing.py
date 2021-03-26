import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_sum
from torch_geometric.nn.inits import reset as reset_modules
from torch_geometric.utils import degree
from torch_geometric.typing import Adj, Size
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Union, List, Dict

from phc.hypercomplex.layers import PHMLinear, PHMMLP
from phc.hypercomplex.norm import PHMNorm
from phc.hypercomplex.utils import phm_cat
from phc.hypercomplex.aggregator import AGGREGATORS, SCALERS, SoftmaxAggregator
from phc.quaternion.activations import get_module_activation


class PHMConv(MessagePassing):
    r"""
    Parametrized Hypercomplex Graphconvolution operator that uses edge-attributes.
    Transformation is a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 learn_phm: True, bias: bool = True,
                 add_self_loops: bool = True,
                 w_init: str = "phm", c_init: str = "standard",
                 aggr: str = "add", same_dim: bool = True,
                 msg_encoder: str = "identity") -> None:
        super(PHMConv, self).__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.phm_rule = phm_rule
        self.learn_phm = learn_phm
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.w_init = w_init
        self.c_init = c_init
        self.aggr = aggr
        self.same_dim = same_dim
        self.transform = PHMLinear(in_features=in_features, out_features=out_features, phm_rule=phm_rule,
                                   phm_dim=phm_dim, bias=bias, w_init=w_init, c_init=c_init,
                                   learn_phm=learn_phm)
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None) -> torch.Tensor:

        if self.add_self_loops:
            x_c = x.clone()
        # propagate messages
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)

        if self.same_dim:
            x = self.transform(x)
            if self.add_self_loops:
                x += x_c
        else:
            if self.add_self_loops:
                x += x_c
            x = self.transform(x)
        return x

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def __repr__(self):
        return "{}(in_features={}, out_features={}, phm_dim={}, phm_rule={}," \
               " learn_phm={}, bias={}, add_self_loops={}, " \
               ", w_init='{}', c_init='{}', aggr='{}')".format(self.__class__.__name__,
                                                self.in_features,
                                                self.out_features,
                                                self.phm_dim,
                                                self.phm_rule,
                                                self.learn_phm,
                                                self.bias,
                                                self.add_self_loops,
                                                self.w_init, self.c_init,
                                                self.aggr)


class PHMGINEConv(MessagePassing):
    r"""
    PHM graph convolution operator similar to the GIN convolution, where the node-features are first
    aggregated and then transformed using a 2-layer PHM MLP.
    """

    def __init__(self, in_features: int, out_features: int, phm_dim: int,
                 phm_rule: Union[None, nn.ParameterList], learn_phm: bool = True,
                 bias: bool = True, add_self_loops: bool = True,
                 norm: [Optional] = None, activation: str = "relu",
                 w_init: str = "phm", c_init: str = "standard",
                 aggr: str = "add",
                 msg_encoder: str = "identity") -> None:
        super(PHMGINEConv, self).__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.learn_phm = learn_phm
        self.phm_rule = phm_rule
        self.norm = norm
        self.activation_str = activation
        self.w_init = w_init
        self.c_init = c_init

        self.aggr = aggr
        self.transform = PHMMLP(in_features=in_features, out_features=out_features,
                                phm_dim=phm_dim, phm_rule=phm_rule,
                                w_init=w_init, c_init=c_init,
                                factor=1, bias=bias, activation=activation, norm=norm)
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)


        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None) -> torch.Tensor:
        if self.add_self_loops:
            x_c = x.clone()
        # propagate messages
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        if self.add_self_loops:
            x += x_c
        # transform
        x = self.transform(x)

        return x

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        assert x_j.size(-1) == edge_attr.size(-1)
        return self.msg_encoder(x_j + edge_attr)

    def __repr__(self):
        return "{}(in_features={}, out_features={}, phm_dim={}, phm_rule={}, learn_phm={}, bias={}, add_self_loops={}, " \
               "norm='{}', activation='{}', w_init='{}',  c_init='{}', aggr='{}')".format(self.__class__.__name__,
                                                                          self.in_features,
                                                                          self.out_features,
                                                                          self.phm_dim,
                                                                          self.phm_rule,
                                                                          self.learn_phm,
                                                                          self.bias,
                                                                          self.add_self_loops,
                                                                          self.norm,
                                                                          self.activation_str,
                                                                          self.w_init, self.c_init,
                                                                          self.aggr)


class PHMConvSoftmax(MessagePassing):
    r"""
    Parametrized Hypercomplex Graphconvolution operator that uses edge-attributes.
    Transformation is a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 learn_phm: True, bias: bool = True,
                 add_self_loops: bool = True,
                 w_init: str = "phm", c_init: str = "standard",
                 aggr: str = "softmax", same_dim: bool = True,
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(PHMConvSoftmax, self).__init__(aggr=None)

        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.phm_rule = phm_rule
        self.learn_phm = learn_phm
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.w_init = w_init
        self.c_init = c_init
        self.aggr = aggr
        self.same_dim = same_dim
        self.transform = PHMLinear(in_features=in_features, out_features=out_features, phm_rule=phm_rule,
                                   phm_dim=phm_dim, bias=bias,
                                   w_init=w_init, c_init=c_init, learn_phm=learn_phm)

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

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None) -> torch.Tensor:

        if self.add_self_loops:
            x_c = x.clone()
        # propagate messages
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)

        if self.same_dim:
            x = self.transform(x)
            if self.add_self_loops:
                x += x_c
        else:
            if self.add_self_loops:
                x += x_c
            x = self.transform(x)
        return x

    def __repr__(self):
        return "{}(in_features={}, out_features={}, phm_dim={}, phm_rule={}," \
               " learn_phm={}, bias={}, add_self_loops={}, " \
               ", w_init='{}', c_init='{}', aggr='{}')".format(self.__class__.__name__,
                                                self.in_features,
                                                self.out_features,
                                                self.phm_dim,
                                                self.phm_rule,
                                                self.learn_phm,
                                                self.bias,
                                                self.add_self_loops,
                                                self.w_init, self.c_init,
                                                self.aggr)


class PHMGINEConvSoftmax(MessagePassing):
    r"""
    PHM graph convolution operator similar to the GIN convolution, where the node-features are first
    aggregated and then transformed using a 2-layer PHM MLP.
    """

    def __init__(self, in_features: int, out_features: int, phm_dim: int,
                 phm_rule: Union[None, nn.ParameterList], learn_phm: bool = True,
                 bias: bool = True, add_self_loops: bool = True,
                 norm: [Optional] = None, activation: str = "relu",
                 w_init: str = "phm", c_init: str = "standard",
                 aggr: str = "softmax",
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(PHMGINEConvSoftmax, self).__init__(aggr=None)

        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.learn_phm = learn_phm
        self.phm_rule = phm_rule
        self.norm = norm
        self.activation_str = activation
        self.w_init = w_init
        self.c_init = c_init
        self.aggr = aggr
        self.transform = PHMMLP(in_features=in_features, out_features=out_features, phm_dim=phm_dim,
                                phm_rule=phm_rule, w_init=w_init, c_init=c_init,
                                factor=1, bias=bias, activation=activation, norm=norm)

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

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None) -> torch.Tensor:
        if self.add_self_loops:
            x_c = x.clone()
        # propagate messages
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        if self.add_self_loops:
            x += x_c
        # transform
        x = self.transform(x)

        return x

    def __repr__(self):
        return "{}(in_features={}, out_features={}, phm_dim={}, phm_rule={}, learn_phm={}, bias={}, add_self_loops={}, " \
               "norm='{}', activation='{}', w_init='{}',  w_init='{}', aggr='{}')".format(self.__class__.__name__,
                                                                          self.in_features,
                                                                          self.out_features,
                                                                          self.phm_dim,
                                                                          self.phm_rule,
                                                                          self.learn_phm,
                                                                          self.bias,
                                                                          self.add_self_loops,
                                                                          self.norm,
                                                                          self.activation_str,
                                                                          self.w_init, self.c_init,
                                                                          self.aggr)



""" 
    Simplified Principal Neighbourhood Aggregation Graph Convolution Operators using hypercomplex multiplication rules
    
    Credits to https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/pna.py
    This version is slightly modified using the hypercomplex linear layer for affine transformations.
"""


class PHMPNAConvSimple(MessagePassing):
    r"""
    Parametrized Hypercomplex Graph Convolutions that utilizes the simple principal neighbourhood aggregation schema.
    """

    def __init__(self, in_features: int, out_features: int,
                 phm_dim: int, phm_rule: Union[None, nn.ParameterList], learn_phm: bool,
                 bias: bool,
                 activation: str, norm: Optional[str],
                 w_init: str, c_init: str,
                 deg: torch.Tensor,
                 aggregators: List[str] = ['mean', 'min', 'max', 'std'],
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 post_layers: int = 1,
                 msg_encoder: str = "relu",
                 **kwargs):

        super(PHMPNAConvSimple, self).__init__(aggr=None, node_dim=0, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        self.activation_str = activation
        self.norm = norm
        self.phm_dim = phm_dim
        self.phm_rule = phm_rule
        self.w_init = w_init
        self.c_init = c_init
        self.learn_phm = learn_phm
        self.aggregators_l = aggregators
        self.scalers_l = scalers
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        self.F_in = in_features
        self.F_out = self.out_features

        self.deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': self.deg.mean().item(),
            'log': (self.deg + 1).log().mean().item(),
            'exp': self.deg.exp().mean().item(),
        }

        in_features = (len(aggregators) * len(scalers)) * self.F_in

        modules = [PHMLinear(in_features=in_features, out_features=self.F_out, bias=self.bias_flag,
                             phm_dim=self.phm_dim, phm_rule=self.phm_rule,
                             w_init=self.w_init, c_init=self.c_init)]
        self.post_layers = post_layers
        for _ in range(post_layers - 1):
            if self.norm:
                modules += [PHMNorm(num_features=self.F_out, phm_dim=self.phm_dim, type="naive-batch-norm")]
            modules += [get_module_activation(self.activation_str)]
            modules += [PHMLinear(in_features=self.F_out, out_features=self.F_out, bias=self.bias_flag,
                                  phm_dim=self.phm_dim, phm_rule=self.phm_rule,
                                  w_init=self.w_init, c_init=self.c_init)]
        self.transform = nn.Sequential(*modules)
        self.msg_encoder_str = msg_encoder
        self.msg_encoder = get_module_activation(activation=msg_encoder)


        self.reset_parameters()

    def reset_parameters(self):
        reset_modules(self)  # thanks to pytorch geometric


    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:

        # x has shape (batch_size, in_feat*phm_dim)
        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, size=size)
        out = self.transform(out)
        return out

    """
    Messages gets called first in PyG's MessagePassing class, see line. 
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py#L237
    """
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        return self.msg_encoder(x_j + edge_attr) if edge_attr is not None else x_j

    """
    Aggregation after messages have been sent, see line.
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py#L253
    """
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  dim_size: Optional[int] = None) -> torch.Tensor:

        # inputs has shape (*, self.phm_dim * self.in_feats)
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        # concatenate the different aggregator results, considering the shape of the hypercomplex components.
        out = phm_cat(tensors=outs, phm_dim=self.phm_dim, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        # concatenate the different aggregator results, considering the shape of the hypercomplex components.
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        out = phm_cat(tensors=outs, phm_dim=self.phm_dim, dim=-1)
        return out

    def __repr__(self):
        return "{}(in_features={}, out_features={}, " \
               "phm_dim={}, learn_phm={}, phm_rule={}, " \
               "bias={}, activation={}, norm={}, w_init={}, c_init={}, " \
               "aggregators={}, scalers={}, " \
               "deg={}, post_layers={})".format(self.__class__.__name__,
                                                self.in_features, self.out_features,
                                                self.phm_dim, self.learn_phm, self.phm_rule,
                                                self.bias_flag, self.activation_str, self.norm, self.w_init, self.c_init,
                                                self.aggregators_l, self.scalers_l, self.deg, self.post_layers)

    def extra_repr(self):
        return self.transform


class PHMMessagePassing(nn.Module):
    r""" Wraps the implemented hypercomplex graph convolution operators into one class """
    def __init__(self, in_features: int, out_features: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 learn_phm: bool = True, bias: bool = True,
                 add_self_loops: bool = True, norm: [Optional] = None, activation: str = "relu",
                 w_init: str = "phm", c_init: str = "standard",
                 aggr: str = "add", mlp: bool = True, same_dim: bool = True,
                 msg_encoder: str = "identity", **kwargs):
        super(PHMMessagePassing, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.norm = norm
        self.learn_phm = learn_phm
        self.phm_rule = phm_rule
        self.activation_str = activation
        self.w_init = w_init
        self.c_init = c_init
        self.aggr = aggr
        self.mlp = mlp
        self.same_dim = same_dim
        self.msg_encoder_str = msg_encoder

        if aggr == "pna":
            self.transform = PHMPNAConvSimple(in_features=in_features, out_features=out_features,
                                              phm_dim=phm_dim, phm_rule=phm_rule, learn_phm=learn_phm,
                                              bias=bias, activation=activation, norm=norm,
                                              w_init=w_init, c_init=c_init,
                                              deg=kwargs.get("deg"),
                                              aggregators=kwargs.get("aggregators"),
                                              scalers=kwargs.get("scalers"),
                                              post_layers=kwargs.get("post_layers"),
                                              msg_encoder="relu")
        elif aggr == "softmax":
            if mlp:
                self.transform = PHMGINEConvSoftmax(in_features, out_features, phm_dim, phm_rule, learn_phm,
                                                    bias, add_self_loops, norm, activation, w_init, c_init,
                                                    aggr, msg_encoder,
                                                    **kwargs)
            else:
                self.transform = PHMConvSoftmax(in_features, out_features, phm_dim, phm_rule, learn_phm,
                                                bias, add_self_loops, w_init, c_init, aggr, same_dim, msg_encoder,
                                                **kwargs)
        else:
            if mlp:
                self.transform = PHMGINEConv(in_features, out_features, phm_dim, phm_rule, learn_phm,
                                             bias, add_self_loops, norm, activation, w_init, c_init, aggr, msg_encoder)
            else:
                self.transform = PHMConv(in_features, out_features, phm_dim, phm_rule, learn_phm,
                                         bias, add_self_loops, w_init, c_init, aggr, same_dim, msg_encoder)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None) -> torch.Tensor:
        return self.transform(x, edge_index, edge_attr, size)

