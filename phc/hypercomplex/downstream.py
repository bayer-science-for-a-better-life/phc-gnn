import torch
import torch.nn as nn
from typing import Union

from phc.hypercomplex.layers import phm_dropout, PHMLinear, RealTransformer
from phc.hypercomplex.norm import PHMNorm

from phc.quaternion.activations import get_module_activation

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

ATOM_FEAT_DIMS = get_atom_feature_dims()
BOND_FEAT_DIMS = get_bond_feature_dims()


""" Hypercomplex Downstream Feed-Forward Network"""


class PHMDownstreamNet(nn.Module):
    """
      A parametrized hypercomplex Feed-Forward Network which predicts a real-valued vector of dimension `out_features`.
    """

    def __init__(self, in_features: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 hidden_layers: list, out_features: int,
                 activation: str, bias: bool, norm: str, w_init: str, c_init: str,
                 dropout: Union[float, list],
                 learn_phm: bool = True, same_dropout: bool = False, real_trafo: str = "linear") -> None:

        super(PHMDownstreamNet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_rule = phm_rule
        self.phm_dim = phm_dim
        self.hidden_layers = hidden_layers
        self.activation_str = activation
        self.activation_func = get_module_activation(activation)
        self.w_init = w_init
        self.c_init = c_init
        self.bias = bias
        self.dropout = [dropout] * len(hidden_layers) if isinstance(dropout, float) else dropout
        assert len(self.dropout) == len(self.hidden_layers), "dropout list must be of the same size " \
                                                             "as number of hidden layer"
        self.norm_type = norm
        self.same_dropout = same_dropout

        # affine linear layers
        # input -> first hidden layer
        self.affine = [PHMLinear(in_features=in_features, phm_dim=self.phm_dim, phm_rule=phm_rule,
                                 out_features=self.hidden_layers[0],
                                 learn_phm=learn_phm, bias=bias, w_init=w_init, c_init=c_init)]
        # hidden layers
        self.affine += [PHMLinear(in_features=self.hidden_layers[i], out_features=self.hidden_layers[i + 1],
                                  phm_dim=self.phm_dim, learn_phm=learn_phm, phm_rule=phm_rule,
                                  bias=bias, w_init=w_init, c_init=c_init)
                        for i in range(len(self.hidden_layers) - 1)]
        # output layer
        self.affine += [PHMLinear(in_features=self.hidden_layers[-1], out_features=self.out_features,
                                  phm_rule=phm_rule,
                                  phm_dim=self.phm_dim, learn_phm=learn_phm, w_init=w_init, c_init=c_init,
                                  bias=bias)]

        self.affine = nn.ModuleList(self.affine)

        # transform the output quaternionic vector to real vector with Real_Transformer module
        self.real_trafo_type = real_trafo
        self.real_trafo = RealTransformer(type=self.real_trafo_type, in_features=self.out_features,
                                          phm_dim=self.phm_dim, bias=True)

        # normalizations
        self.norm_flag = False
        if self.norm_type:
            norm_type = self.norm_type
            self.norm = [PHMNorm(num_features=dim, phm_dim=self.phm_dim, type=norm_type) for dim in self.hidden_layers]
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

    def forward(self, x: torch.Tensor, verbose=False, **kwargs) -> torch.Tensor:
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
                    x = self.norm[i](x, **kwargs)
                    x = self.activation_func(x)
                else:
                    if verbose:
                        print("activation")
                    x = self.activation_func(x)
                if self.training and self.dropout[i] > 0.0:  # and i > 0:
                    if verbose:
                        print("dropout")
                    x = phm_dropout(x, p=self.dropout[i], phm_dim=self.phm_dim,
                                    training=self.training, same=self.same_dropout)
            if verbose:
                print("output:", x.size())
        # at the end, transform the quaternion output vector to a real vector
        x = self.real_trafo(x)
        return x

    def __repr__(self):
        return "{}(in_features={}, phm_dim={}, phm_rule={}, hidden_layers={}, out_features={}, \n" \
               "activation='{}', bias={}, norm='{}', init='{}', dropout={}, learn_phm={}, \n" \
               "same_dropout={}, real_trafo='{}')".format(self.__class__.__name__,
                                                          self.in_features, self.phm_dim, self.phm_rule,
                                                          self.hidden_layers, self.out_features,
                                                          self.activation_str, self.bias, self.norm_type, self.init,
                                                          self.dropout, self.learn_phm,
                                                          self.same_dropout, self.real_trafo_type)


"""
model = PHMDownstreamNet(in_features=64, phm_dim=5, hidden_layers=[32, 16], out_features=1, activation="relu",
                         bias=True, norm="naive-batch-norm", init="phm", dropout=[0.1, 0.2],
                         same_dropout=False, real_trafo="linear")

x = torch.randn(512, 64*5)
y = model(x, verbose=True)
"""