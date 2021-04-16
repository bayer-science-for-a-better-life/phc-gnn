import torch
import torch.nn.functional as F
import torch.nn as nn

from phc.hypercomplex.kronecker import kronecker_product, kronecker_product_einsum_batched
from phc.hypercomplex.utils import get_multiplication_matrices, right_cyclic_permutation
from phc.hypercomplex.inits import phm_init
from phc.hypercomplex.norm import PHMNorm

from phc.quaternion.inits import glorot_uniform, glorot_normal
from phc.quaternion.activations import get_module_activation

from typing import Optional, Union


def get_bernoulli_mask(x: torch.Tensor, p: float) -> torch.Tensor:
    mask = torch.empty(x.size(), device=x.device).fill_(1-p)
    mask = torch.bernoulli(mask)
    return mask


def torch_dropout(x: torch.Tensor, p: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        mask = get_bernoulli_mask(x, p)
    dropped = mask * x
    # scale by 1/(1-p) as in https://jmlr.org/papers/v15/srivastava14a.html
    dropped = 1/(1-p) * dropped
    return dropped


def phm_dropout(x: torch.Tensor, phm_dim: int,
                p: float = 0.2, training: bool = True, same: bool = False) -> torch.Tensor:
    assert 0.0 <= p <= 1.0, f"dropout rate must be in [0.0 ; 1.0]. {p} was inserted!"
    r"""
    Applies the same dropout mask for each phm component tensor of size [num_batch_nodes, d*in_feat]
    along the same dimension in_feat for the hypercomplex components d.
    :param x: phm tensor of size [d, num_batch_nodes, in_feat]
    :param p: dropout rate. Must be within [0.0 ; 1.0]. If p=0.0, this function returns the input tensors 
    :param training: boolean flag if the dropout is used in training mode
                     Only if this is True, the dropout will be applied. Otherwise it will return the input tensors
    :return: (droped-out) phm tensor q
    """
    if training and p > 0.0:
        if same:
            in_feats_per_component = x.size(-1) // phm_dim
            # (phm_dim, batch_size, in_feats)
            x = x.reshape(x.size(0), phm_dim, in_feats_per_component).permute(1, 0, 2)
            mask = get_bernoulli_mask(x[0], p=p)
            mask = mask.unsqueeze(dim=0)
            x = torch_dropout(x=x, p=p, mask=mask)
            x = x.permute(1, 0, 2)
            x = x.reshape(x.size(0), -1)  # (batch_size, phm_dim*in_feat)
        else:
            x = F.dropout(input=x, p=p, training=training)
    return x


def matvec_product(W: nn.ParameterList, x: torch.Tensor,
                   bias: Optional[nn.ParameterList],
                   phm_rule: Union[list, nn.ParameterList]) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    
    W is a nn.ParamterList with len(phm_rule) tensors of size (out_features, in_features)
    x has shape (batch_size, phm_dim*in_features)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product

    As of now, it iterates over the "hyper-imaginary" components, a more efficient implementation
    would be to stack the x and bias vector directly as a 1D vector.
    """
    assert len(phm_rule) == len(W)
    assert x.size(-1) == sum([weight.size(1) for weight in W]), (f"x has size(1): {x.size(-1)}."
                                                                f"Should have {sum([weight.size(1) for weight in W])}")

    #H = torch.stack([kronecker_product(Ai, Wi) for Ai, Wi in zip(phm_rule, W)], dim=0).sum(0)
    A = torch.stack([Ai for Ai in phm_rule], dim=0)
    W = torch.stack([Wi for Wi in W], dim=0)
    H = kronecker_product(A, W).sum(0)

    #y = torch.mm(H, x.t()).t()
    y = torch.matmul(input=x, other=H.t())
    #y = (H @ x.T).T
    if bias is not None:
        bias = torch.cat([b for b in bias], dim=-1)
        y += bias
    return y



def matvec_product_cat(W: nn.ParameterList, x: torch.Tensor,
                       bias: Optional[nn.ParameterList], phm_rule: Union[list, nn.ParameterList]) -> torch.Tensor:
    assert len(phm_rule) == len(W)
    assert x.size(1) == sum([weight.size(1) for weight in W])

    A_mat = torch.stack([A for A in phm_rule], dim=0).sum(0)

    ids = list(range(len(phm_rule)))
    H = []
    for i, contribution in enumerate(A_mat):
        permuted = right_cyclic_permutation(ids, power=i)
        concatenated_weights = torch.cat([contribution[i] * W[p] for i, p in enumerate(permuted)], dim=-1)
        H.append(concatenated_weights)
    H = torch.cat(H, dim=0)
    y = torch.mm(H, x.t()).t()
    if bias is not None:
        bias = torch.cat([b for b in bias], dim=-1)
        y += bias
    return y


class PHMLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 phm_dim: int, phm_rule: Union[None, nn.Parameter, nn.ParameterList, list, torch.Tensor] = None,
                 bias: bool = True, w_init: str = "phm", c_init: str = "standard",
                 learn_phm: bool = True) -> None:
        super(PHMLinear, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform"]
        assert c_init in ["standard", "random"]
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim

        self.shared_phm = False
        if phm_rule is not None:
            self.shared_phm = True
            self.phm_rule = phm_rule
            if not isinstance(phm_rule, nn.ParameterList) and learn_phm:
                self.phm_rule = nn.ParameterList([nn.Parameter(mat, requires_grad=learn_phm) for mat in self.phm_rule])
        else:
            self.phm_rule = get_multiplication_matrices(phm_dim, type=c_init)

        self.phm_rule = nn.ParameterList([nn.Parameter(mat, requires_grad=learn_phm) for mat in self.phm_rule])

        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features),
                                                requires_grad=True)
                                   for _ in range(phm_dim)])
        if self.bias_flag:
            self.b = nn.ParameterList(
                [nn.Parameter(torch.Tensor(out_features), requires_grad=True) for _ in range(phm_dim)]
            )
        else:
            self.register_parameter("b", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.w_init == "phm":
            W_init = phm_init(phm_dim=self.phm_dim, in_features=self.in_features, out_features=self.out_features)
            for W_param, W_i in zip(self.W, W_init):
                W_param.data = W_i.data

        elif self.w_init == "glorot-normal":
            for i in range(self.phm_dim):
                self.W[i] = glorot_normal(self.W[i])
        elif self.w_init == "glorot-uniform":
            for i in range(self.phm_dim):
                self.W[i] = glorot_uniform(self.W[i])
        else:
            raise ValueError
        if self.bias_flag:
            self.b[0].data.fill_(0.0)
            for bias in self.b[1:]:
                bias.data.fill_(0.2)

        if not self.shared_phm:
            phm_rule = get_multiplication_matrices(phm_dim=self.phm_dim, type=self.c_init)
            for i, init_data in enumerate(phm_rule):
                self.phm_rule[i].data = init_data

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
        # #ToDo modify forward() functional so it can handle shared phm-rule contribution matrices.
        return matvec_product(W=self.W, x=x, bias=self.b, phm_rule=self.phm_rule)

    def __repr__(self):
        return '{}(in_features={}, out_features={}, ' \
               'phm_dim={}, ' \
               'bias={}, w_init={}, c_init={}, ' \
               'learn_phm={})'.format(self.__class__.__name__,
                                      self.in_features,
                                      self.out_features,
                                      self.phm_dim,
                                      self.bias_flag,
                                      self.w_init,
                                      self.c_init,
                                      self.learn_phm)


"""  Improving PHMLinear  """


def matvec_product_new(W: torch.Tensor, x: torch.Tensor,
                       bias: Optional[torch.Tensor],
                       phm_rule: Union[torch.Tensor]) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.

    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product

    """

    H = kronecker_product_einsum_batched(phm_rule, W).sum(0)
    y = torch.matmul(input=x, other=H)
    if bias is not None:
        y += bias

    return y


class PHMLinearNew(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 phm_dim: int, phm_rule: Union[None, torch.Tensor] = None,
                 bias: bool = True, w_init: str = "phm", c_init: str = "random",
                 learn_phm: bool = True) -> None:
        super(PHMLinearNew, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform"]
        assert c_init in ["standard", "random"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"

        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim

        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim

        if phm_rule is not None:
            self.phm_rule = phm_rule
        else:
            self.phm_rule = get_multiplication_matrices(phm_dim, type=c_init)

        self.phm_rule = nn.Parameter(torch.stack([*self.phm_rule], dim=0), requires_grad=learn_phm)

        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.W = nn.Parameter(torch.Tensor((phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)),
                              requires_grad=True)
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.w_init == "phm":
            W_init = phm_init(phm_dim=self.phm_dim,
                              in_features=self._in_feats_per_axis,
                              out_features=self._out_feats_per_axis, transpose=False)
            self.W.data = W_init

        elif self.w_init == "glorot-normal":
            for i in range(self.phm_dim):
                self.W[i] = glorot_normal(self.W[i])
        elif self.w_init == "glorot-uniform":
            for i in range(self.phm_dim):
                self.W[i] = glorot_uniform(self.W[i])
        else:
            raise ValueError
        if self.bias_flag:
            self.b.data[:self._out_feats_per_axis] = 0.0
            self.b.data[(self._out_feats_per_axis+1):] = 0.2

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
        # #ToDo modify forward() functional so it can handle shared phm-rule contribution matrices.
        return matvec_product_new(W=self.W, x=x, bias=self.b, phm_rule=self.phm_rule)

    def __repr__(self):
        return '{}(in_features={}, out_features={}, ' \
               'phm_dim={}, ' \
               'bias={}, w_init={}, c_init={}, ' \
               'learn_phm={})'.format(self.__class__.__name__,
                                      self.in_features,
                                      self.out_features,
                                      self.phm_dim,
                                      self.bias_flag,
                                      self.w_init,
                                      self.c_init,
                                      self.learn_phm)


""""""

class PHMMLP(nn.Module):
    """ Implementing a 2-layer PHM Multilayer Perceptron """
    def __init__(self, in_features: int, out_features: int, phm_dim: int, phm_rule: Union[None, nn.ParameterList],
                 bias: bool = True,
                 learn_phm: bool = True,
                 activation: str = "relu", norm: Union[None, str] = None,
                 w_init: str = "phm", c_init: str = "standard",
                 factor: float = 1, **kwargs) -> None:

        super(PHMMLP, self).__init__()
        assert norm in ["None", None, "naive-batch-norm", "naive-naive-batch-norm"]
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.bias_flag = bias
        self.learn_phm = learn_phm
        self.phm_rule = phm_rule
        self.activation_str = activation
        self.linear1 = PHMLinear(in_features=in_features, out_features=int(factor*out_features),
                                 phm_dim=phm_dim, phm_rule=phm_rule, learn_phm=learn_phm, bias=bias,
                                 w_init=w_init, c_init=c_init)
        self.linear2 = PHMLinear(in_features=int(factor*out_features), out_features=out_features,
                                 phm_dim=phm_dim, phm_rule=phm_rule, learn_phm=learn_phm, bias=bias,
                                 w_init=w_init, c_init=c_init)
        self.activation = get_module_activation(activation)
        self.norm_type = norm
        self.factor = factor
        self.w_init = w_init
        self.c_init = c_init
        if norm in ["naive-batch-norm", "naive-naive-batch-norm"]:
            self.norm_flag = True
            self.norm = PHMNorm(num_features=int(factor*out_features), phm_dim=self.phm_dim,
                                type=norm, **kwargs)
        else:
            self.norm_flag = False

        self.reset_parameters()


    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        if self.norm_flag:
            self.norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.norm_flag:
            x = self.norm(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    def __repr__(self):
        return '{}(in_features={}, out_features={}, phm_dim={}, phm_rule={}, bias={}, ' \
               'learn_phm={}, activation="{}", norm="{}", ' \
               'w_init="{}", c_init={}, factor={})'.format(self.__class__.__name__,
                                                           self.in_features,
                                                           self.out_features,
                                                           self.phm_dim,
                                                           self.phm_rule,
                                                           self.bias_flag,
                                                           self.learn_phm,
                                                           self.activation_str,
                                                           self.norm_type, self.w_init, self.c_init,
                                                           self.factor)


class RealTransformer(torch.nn.Module):
    def __init__(self, type: str, in_features: int, phm_dim: int, bias: bool = True) -> None:
        """
        Initializes a Real transofmation layer that
        converts a hypercomplex vector \mathbb{H}^d to a real vector \mathbb{R}^d
        Here d equals to `in_features`.
        """
        super(RealTransformer, self).__init__()
        assert type in ["linear", "sum", "mean", "norm"]
        self.type = type
        self.in_features = in_features
        self.phm_dim = phm_dim
        self.bias_flag = bias
        if self.type == "linear":
            self.affine = torch.nn.Linear(in_features=phm_dim * self.in_features,
                                          out_features=self.in_features, bias=bias)
        else:
            self.affine = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.type == "linear":
            torch.nn.init.xavier_uniform_(self.affine.weight)
            if self.bias_flag:
                self.affine.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass to convert a hypercomplex vector to a real vector.
        Here, x consists of the already concatenated features, i.e. x has shape (batch_size, phm_dim * feats)
        """
        if self.type in ["sum", "mean", "norm"]:
            x = torch.stack(*[x.split(split_size=self.in_features, dim=-1)], dim=0)

        if self.type == "sum":
            return x.sum(dim=0)
        elif self.type == "mean":
            return x.mean(dim=0)
        elif self.type == "norm":
            return x.norm(dim=0)
        else:
            return self.affine(x)

    def __repr__(self):
        return '{}(type="{}", in_features={}, phm_dim={}, bias={})'.format(self.__class__.__name__,
                                                                           self.type,
                                                                           self.in_features,
                                                                           self.phm_dim, self.bias_flag)


"""
phm_dim = 4
in_features = 32
out_features= 64
batch_size = 512
x = torch.randn(batch_size, in_features*phm_dim)
linear_layer = PHMLinear(in_features=in_features, out_features=out_features,
                         phm_dim=phm_dim, w_init="phm", c_init="random", bias=True, learn_phm=True)
y = linear_layer(x)

real_trafo = RealTransformer(type="linear", in_features=out_features, phm_dim=phm_dim)
yy = real_trafo(y)

real_trafo = RealTransformer(type="sum", in_features=out_features, phm_dim=phm_dim)
yyy = real_trafo(y)
"""
