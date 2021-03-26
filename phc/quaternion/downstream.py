import torch
import torch.nn as nn
from typing import Union

from phc.quaternion.algebra import QTensor
from phc.quaternion.activations import get_functional_activation
from phc.quaternion.layers import quaternion_dropout, QLinear, RealTransformer
from phc.quaternion.norm import QuaternionNorm

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

ATOM_FEAT_DIMS = get_atom_feature_dims()
BOND_FEAT_DIMS = get_bond_feature_dims()


""" Quaternion Downstream Feed-Forward Network"""


class QuaternionDownstreamNet(nn.Module):
    """  A quaternion Feed-Forward Network which predicts a real-valued vector of dimension `out_features`. """

    def __init__(self, in_features: int, hidden_layers: list, out_features: int,
                 activation: str, bias: bool, norm: str, init: str,
                 dropout: Union[float, list], same_dropout: bool = False, real_trafo: str = "linear") -> None:

        super(QuaternionDownstreamNet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.activation_str = activation
        self.activation_func = get_functional_activation(activation)
        self.init = init
        self.bias = bias
        self.dropout = [dropout] * len(hidden_layers) if isinstance(dropout, float) else dropout
        assert len(self.dropout) == len(self.hidden_layers), "dropout list must be of the same size " \
                                                             "as number of hidden layer"
        self.norm_type = norm
        self.same_dropout = same_dropout

        # affine linear layers
        # input -> first hidden layer
        self.affine = [QLinear(in_features, self.hidden_layers[0], bias=bias, init=init)]
        # hidden layers
        self.affine += [QLinear(self.hidden_layers[i], self.hidden_layers[i + 1], bias=bias, init=init)
                        for i in range(len(self.hidden_layers) - 1)]
        # output layer
        self.affine += [QLinear(self.hidden_layers[-1], self.out_features, init=init, bias=bias)]

        self.affine = nn.ModuleList(self.affine)

        # transform the output quaternionic vector to real vector with Real_Transformer module
        self.real_trafo_type = real_trafo
        self.real_trafo = RealTransformer(type=self.real_trafo_type, in_features=self.out_features, bias=True)

        # normalizations
        self.norm_flag = False
        if "norm" in self.norm_type:
            norm_type = self.norm_type
            self.norm = [QuaternionNorm(num_features=dim, type=norm_type) for dim in self.hidden_layers]
            self.norm = nn.ModuleList(self.norm)
            self.norm_flag = True

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.affine:
            module.reset_parameters()
        if self.norm_flag:
            for module in self.norm:
                module.reset_parameters()
        self.real_trafo.reset_parameters()

    def forward(self, x: QTensor, verbose=False) -> torch.Tensor:
        # forward pass
        for i in range(len(self.affine)):

            if verbose:
                print(f"iteration {i}")
                print("input:", x.size())
                print("affine", self.affine[i])
            x = self.affine[i](x)
            # print("out affine:", x.size())
            if i < len(self.affine) - 1:  # only for input->hidden and hidden layers, but not output
                if self.norm_flag:
                    if verbose:
                        print("normalization")
                        print("activation")
                    x = self.norm[i](x)
                    x = self.activation_func(x)
                else:
                    if verbose:
                        print("activation")
                    x = self.activation_func(x)
                if self.training and self.dropout[i] > 0.0:  # and i > 0:
                    if verbose:
                        print("dropout")
                    x = quaternion_dropout(x, p=self.dropout[i], training=self.training, same=self.same_dropout)
            if verbose:
                print("output:", x.size())
        # at the end, transform the quaternion output vector to a real vector
        x = self.real_trafo(x)
        return x

    def __repr__(self):
        return "{}(in_features={}, hidden_layers={}, out_features={}, \n" \
               "activation='{}', bias={}, norm='{}', init='{}', dropout={}, \n" \
               "same_dropout={}, real_trafo='{}')".format(self.__class__.__name__,
                                                          self.in_features, self.hidden_layers, self.out_features,
                                                          self.activation_str, self.bias, self.norm_type, self.init,
                                                          self.dropout, self.same_dropout, self.real_trafo_type)

"""
model = QuaternionDownstreamNet(in_features=64, hidden_layers=[32, 16], out_features=1, activation="relu",
                                bias=True, norm="naive-batch-norm", init="orthogonal", dropout=[0.1, 0.2],
                                same_dropout=False, real_trafo="linear")

x = QTensor(*torch.randn(4, 256, 64))
y = model(x, verbose=True)
"""
