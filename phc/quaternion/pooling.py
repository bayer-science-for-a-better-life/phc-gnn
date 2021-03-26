import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch

from phc.quaternion.layers import QLinear, RealTransformer
from phc.quaternion.algebra import QTensor



class QuaternionGlobalSumPooling(nn.Module):
    def __init__(self):
        super(QuaternionGlobalSumPooling, self).__init__()
        self.module = global_add_pool

    def __call__(self, x: QTensor, batch: Batch) -> QTensor:
        x_tensor = x.stack(dim=1)  # transform to torch.Tensor
        pooled = self.module(x=x_tensor, batch=batch)  # apply global pooling
        pooled = pooled.permute(1, 0, 2)  # permute such that first dimension is (4,*,*)
        return QTensor(*pooled)

    def reset_parameters(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"



class QuaternionSoftAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = True, init: str = "orthogonal",
                 real_trafo: str = "linear"):
        super(QuaternionSoftAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.init = init
        self.real_trafo_type = real_trafo
        self.bias = bias
        self.linear = QLinear(in_features=self.embed_dim, out_features=self.embed_dim,
                              init=init, bias=bias)
        self.real_trafo = RealTransformer(type=self.real_trafo_type, in_features=self.embed_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.sum_pooling = QuaternionGlobalSumPooling()
        self.reset_parameters()

    def reset_parameters(self):
        self.real_trafo.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x: QTensor, batch: Batch) -> QTensor:
        out = self.linear(x)  # get logits
        out = self.real_trafo(out)  # "transform" to real-valued
        out = self.sigmoid(out)  # get "probabilities"
        x = QTensor(out * x.r, out * x.i, out * x.j, out * x.k)   # explicitly writing out the hadamard product
        x = self.sum_pooling(x, batch)
        return x

    def __repr__(self):
        return "{}(embed_dim={}, bias={}, init='{}', real_trafo='{}')".format(self.__class__.__name__,
                                                                              self.embed_dim,
                                                                              self.bias,
                                                                              self.init,
                                                                              self.real_trafo_type)
