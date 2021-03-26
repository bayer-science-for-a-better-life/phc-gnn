import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional

from phc.quaternion.algebra import QTensor, hamilton_product_Wq
from phc.quaternion.inits import quaternion_init, glorot_normal, glorot_uniform, orthogonal_init
from phc.quaternion.activations import get_functional_activation
from phc.quaternion.norm import QuaternionNorm


def get_bernoulli_mask(x: torch.Tensor, p: float) -> torch.Tensor:
    mask = torch.empty(x.size(), device=x.device).fill_(1-p)
    mask = torch.bernoulli(mask)
    return mask


def torch_dropout(x: torch.Tensor, p: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        mask = get_bernoulli_mask(x, p)
    dropped = mask * x
    # scale by 1/(1-p) as done by pytorch https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    dropped /= (1-p)
    return dropped


def quaternion_dropout(q: QTensor, p: float = 0.2, training: bool = True, same: bool = False) -> QTensor:
    assert 0.0 <= p <= 1.0, f"dropout rate must be in [0.0 ; 1.0]. {p} was inserted!"
    r"""
    Applies the same dropout mask for each quaternion component tensor of size [num_batch_nodes, d]
    along the same dimension d for the real and three hypercomplex parts.
    :param q: quaternion tensor with real part r and three hypercomplex parts i,j,k
    :param p: dropout rate. Must be within [0.0 ; 1.0]. If p=0.0, this function returns the input tensors 
    :param training: boolean flag if the dropout is used in training mode
                     Only if this is True, the dropout will be applied. Otherwise it will return the input tensors
    :return: (droped-out) quaternion q
    """
    if training and p > 0.0:
        q = q.stack(dim=0)
        if same:
            mask = get_bernoulli_mask(x=q[0], p=p).unsqueeze(dim=0)
            q = torch_dropout(x=q, p=p, mask=mask)
        else:
            q = F.dropout(q, p=p, training=training)
        return QTensor(*q)
    else:
        return q


class QLinear(nn.Module):
    """ Quaternion Linear Layer that applies the affine transformation h = f(x) = Wx + b """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = "orthogonal") -> None:
        # Initialize empty weight matrices for real part (r) and three complex parts (i,j,k)
        super(QLinear, self).__init__()
        assert init in ["glorot-normal", "glorot-uniform", "quaternion", "orthogonal"]
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init = init
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.W_i = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.W_j = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)

        if self.bias:
            self.b_r = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
            self.b_i = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
            self.b_j = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
            self.b_k = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)

        self.reset_parameters()

        # save the weights into QTensors
        self.W = QTensor(self.W_r, self.W_i, self.W_j, self.W_k)
        if self.bias:
            self.b = QTensor(self.b_r, self.b_i, self.b_j, self.b_k)

    def reset_parameters(self):
        if self.init == "quaternion":
            # just fill data attributes
            self.W_r.data, self.W_i.data, self.W_j.data, self.W_k.data = quaternion_init(in_features=self.in_features,
                                                                                         out_features=self.out_features,
                                                                                         low=0, high=1, transpose=True)

        elif self.init == "orthogonal":
            self.W_r.data, self.W_i.data, self.W_j.data, self.W_k.data = orthogonal_init(in_features=self.in_features,
                                                                                         out_features=self.out_features,
                                                                                         transpose=True)

        elif self.init == "glorot-normal":
            self.W_r = glorot_normal(self.W_r)
            self.W_i = glorot_normal(self.W_i)
            self.W_j = glorot_normal(self.W_j)
            self.W_k = glorot_normal(self.W_k)

        elif self.init == "glorot-uniform":
            self.W_r = glorot_uniform(self.W_r)
            self.W_i = glorot_uniform(self.W_i)
            self.W_j = glorot_uniform(self.W_j)
            self.W_k = glorot_uniform(self.W_k)

        if self.bias:
            # initialise the biases as 0 for the real component and 0.2 for the others
            self.b_r.data.fill_(value=0)
            self.b_i.data.fill_(value=0.2)
            self.b_j.data.fill_(value=0.2)
            self.b_k.data.fill_(value=0.2)


    def forward(self, q: QTensor, **kwargs) -> QTensor:
        """ Applies the affine transformation h = Wx + b where W,x and b are quaternion-valued. """
        q = hamilton_product_Wq(W=self.W, q=q)

        if self.bias:
            q = q + self.b

        return q

    def __repr__(self):
        return '{}(in_features={}, out_features={}, ' \
               'bias={}, init={})'.format(self.__class__.__name__,
                                          self.in_features,
                                          self.out_features,
                                          self.bias,
                                          self.init)


class QMLP(nn.Module):
    """ Implementing a 2-layer Quaternion Multilayer Perceptron """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = "relu", norm: Union[None, str] = None,
                 init: str = "orthogonal", factor: float = 1, **kwargs) -> None:

        super(QMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_str = activation
        self.qlinear1 = QLinear(in_features=in_features, out_features=int(factor*out_features),
                                bias=bias, init=init)
        self.qlinear2 = QLinear(in_features=int(factor*out_features), out_features=out_features,
                                bias=bias, init=init)
        self.activation = get_functional_activation(activation)
        self.norm_type = norm
        self.factor = factor
        self.init_type = init
        if norm in ["naive-batch-norm", "q-batch-norm"]:
            self.norm_flag = True
            self.norm = QuaternionNorm(num_features=int(factor*out_features), type=norm, **kwargs)
        else:
            self.norm_flag = False

        self.reset_parameters()


    def reset_parameters(self):
        self.qlinear1.reset_parameters()
        self.qlinear2.reset_parameters()
        if self.norm_flag:
            self.norm.reset_parameters()

    def forward(self, q: QTensor):
        q = self.qlinear1(q)
        if self.norm_flag:
            q = self.norm(q)
        q = self.activation(q)
        q = self.qlinear2(q)
        return q

    def __repr__(self):
        return '{}(in_features={}, out_features={}, bias={}, ' \
               'activation="{}", norm="{}", ' \
               'init="{}", factor={})'.format(self.__class__.__name__,
                                              self.in_features,
                                              self.out_features,
                                              self.bias,
                                              self.activation_str,
                                              self.norm_type, self.init_type,
                                              self.factor)


class RealTransformer(torch.nn.Module):
    def __init__(self, type: str, in_features: int, bias: bool = True) -> None:
        """
        Initializes a Real transofmation layer that
        converts a quaternion vector \mathbb{H}^d to a real vector \mathbb{R}^d
        Here d equals to `in_features`.
        """
        super(RealTransformer, self).__init__()
        assert type in ["linear", "sum", "mean", "norm"]
        self.type = type
        self.in_features = in_features
        self.bias = bias
        if self.type == "linear":
            self.affine = torch.nn.Linear(in_features=4 * self.in_features, out_features=self.in_features, bias=bias)
        else:
            self.affine = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.type == "linear":
            torch.nn.init.xavier_uniform_(self.affine.weight)
            if self.bias:
                self.affine.bias.data.fill_(0.0)

    def forward(self, x: QTensor) -> torch.Tensor:
        """Computes the forward pass to convert a quaternion vector to a real vector"""
        if self.type == "sum":
            return x.r + x.i + x.j + x.k
        elif self.type == "mean":
            return (x.r + x.i + x.j + x.k).mean()
        elif self.type == "norm":
            return x.norm()
        else:
            x = torch.cat([x.r, x.i, x.j, x.k], dim=-1)
            return self.affine(x)

    def __repr__(self):
        return '{}(type="{}", in_features={}, bias={})'.format(self.__class__.__name__,
                                                               self.type,
                                                               self.in_features, self.bias)


class QLayerBlock(nn.Module):
    """
    A quaternion hidden layer block
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, init: str,
                 activation: str, norm: Optional[str]) -> None:
        super(QLayerBlock, self).__init__()
        assert norm in [None, "naive-batch-norm", "q-batch-norm"]
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init = init
        self.activation_str = activation
        self.activation = get_functional_activation(activation)
        self.norm = QuaternionNorm(num_features=out_features, type=norm) if norm is not None else None
        self.affine = QLinear(in_features=in_features, out_features=out_features, bias=bias, init=init)

        self.reset_parameters()

    def reset_parameters(self):
        self.affine.reset_parameters()

    def forward(self, x: QTensor, verbose: False) -> QTensor:
        if verbose:
            print("affine trafo")
        x = self.affine(x)
        if self.norm is not None:
            if verbose:
                print("normalization")
            x = self.norm(x)
        if verbose:
            print("activation")
        x = self.activation(x)
        return x

    def __repr__(self):
        return "{}(in_features={}, out_features={}, bias={} \n" \
               "init='{}', activation={}, norm='{}')".format(self.__class__.__name__,
                                                             self.in_features, self.out_features, self.bias,
                                                             self.init, self.activation, self.norm)



class QFNN(nn.Module):
    """
    A simple quaternion feed-forward network.
    """
    def __init__(self, input_dim: int, hidden_dims: Union[int, list], output_dim: int, activation: str, init: str,
                 bias: bool, norm: Optional[str], dropout: Union[float, list], same_dropout: bool = False) -> None:

        super(QFNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.output_dim = output_dim
        self.activation_str = activation
        self.init = init
        self.bias = bias
        self.norm = norm
        self.dropout = [dropout] if isinstance(dropout, int) else dropout
        assert len(self.hidden_dims) == len(self.dropout)
        self.same_dropout = same_dropout

        # input -> 1st hidden layer
        layers = [QLayerBlock(in_features=input_dim, out_features=self.hidden_dims[0], activation=activation,
                              bias=bias, norm=norm, init=init)]

        # hidden layers
        for i in range(len(hidden_dims) - 1):
            layers += [QLayerBlock(in_features=self.hidden_dims[i], out_features=self.hidden_dims[i+1],
                                   activation=activation, bias=bias, norm=norm, init=init)
                       ]

        # output layer
        layers += [QLinear(in_features=self.hidden_dims[-1], out_features=output_dim, bias=bias, init=init)]

        self.layers = nn.ModuleList(layers)

        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, x: QTensor, verbose: bool = False) -> QTensor:
        # forward pass from input layer to output layer
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"iteration: {i}")
                print("- layer block -")
            x = layer(x, verbose=verbose)
            if i < len(self.layers) - 1:  # only dropout after 1st hidden layer and before output layer
                if verbose:
                    print("dropout")
                x = quaternion_dropout(q=x, p=self.dropout[i], training=self.training, same=self.same_dropout)
        return x


    def __repr__(self):
        return '{}(input_dim={}, hidden_dims={}, output_dim={},' \
               'activation=\'{}\', bias={}, norm=\'{}\', init=\'{}\', dropout={},' \
               'same_dropout={})'.format(self.__class__.__name__, self.input_dim, self.hidden_dims, self.output_dim,
                                         self.activation_str, self.bias, self.norm, self.init, self.dropout,
                                         self.same_dropout)

"""
x = QTensor(*torch.randn(4, 512, 128))
model = QFNN(input_dim=128, hidden_dims=[128, 64, 32], output_dim=1, activation="relu", init="orthogonal", bias=True,
             norm="naive-batch-norm", dropout=[0.1, 0.2, 0.0], same_dropout=False)

y = model(x, verbose=True)

"""