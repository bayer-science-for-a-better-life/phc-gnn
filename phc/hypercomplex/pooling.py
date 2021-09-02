import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from typing import Union

from phc.hypercomplex.layers import PHMLinear, RealTransformer


class PHMGlobalSumPooling(nn.Module):
    def __init__(self, phm_dim: int):
        super(PHMGlobalSumPooling, self).__init__()
        self.phm_dim = phm_dim
        self.module = global_add_pool

    def __call__(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        # x has shape (batch_num_nodes, self.phm_dim * in_feats)
        x = self.module(x=x, batch=batch)  # apply global pooling
        return x

    def reset_parameters(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(phm_dim={self.phm_dim})"



class PHMSoftAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 learn_phm: bool = True,
                 bias: bool = True, w_init: str = "phm", c_init: str = "standard",
                 real_trafo: str = "linear"):
        super(PHMSoftAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.phm_dim = phm_dim
        self.w_init = w_init
        self.c_init = c_init
        self.phm_rule = phm_rule
        self.learn_phm = learn_phm
        self.real_trafo_type = real_trafo
        self.bias = bias
        self.linear = PHMLinear(in_features=self.embed_dim, out_features=self.embed_dim, phm_dim=self.phm_dim,
                                phm_rule=phm_rule,
                                learn_phm=learn_phm, w_init=w_init, c_init=c_init,
                                bias=bias)
        self.real_trafo = RealTransformer(type=self.real_trafo_type, phm_dim=self.phm_dim,
                                          in_features=self.embed_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.sum_pooling = PHMGlobalSumPooling(phm_dim=self.phm_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.real_trafo.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        out = self.linear(x)  # get logits
        print(out.shape)
        out = self.real_trafo(out)  # "transform" to real-valued
        print(out.shape)
        out = self.sigmoid(out)  # get "probabilities"
        #x = torch.stack([*x.split(split_size=self.embed_dim, dim=-1)], dim=0)
        x = x.reshape(x.size(0), self.phm_dim, self.embed_dim // self.phm_dim)
        print(x.shape)
        # apply element-wise hadamard product through broadcasting
        out = out.unsqueeze(dim=1)
        x = out * x
        x = x.reshape(x.size(0), self.embed_dim)
        x = self.sum_pooling(x, batch=batch)
        return x

    def __repr__(self):
        return "{}(embed_dim={}, phm_dim={}, phm_rule={}, learn_phm={}," \
               "bias={}, init='{}', real_trafo='{}')".format(self.__class__.__name__,
                                                             self.embed_dim,
                                                             self.phm_dim,
                                                             self.phm_rule,
                                                             self.learn_phm,
                                                             self.bias,
                                                             self.init,
                                                             self.real_trafo_type)