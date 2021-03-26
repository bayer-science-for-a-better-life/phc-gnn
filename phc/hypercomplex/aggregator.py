import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.utils import degree

from typing import Optional, Dict, List, Optional

from phc.hypercomplex.utils import phm_cat
from phc.hypercomplex.layers import PHMLinear



class Aggregator(nn.Module):
    """ Naive aggregator that inherits from nn.Module in case we want a parametrized version. """
    def __init__(self, reduce: str = "sum", phm_dim: int = 4) -> None:
        super(Aggregator, self).__init__()
        assert reduce in ["sum", "min", "max", "mean"]
        self.reduce = reduce
        self.phm_dim = phm_dim

    def reset_parameters(self):
        pass

    def forward(self, x: torch.Tensor, idx: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> torch.Tensor:
        r"""
        Gathers tensor-values together based on idx tensor.
        x is tensor of size (batch_num_nodes, self.phm_dim * feats).
        """
        aggr = torch_scatter.scatter(src=x, index=idx, dim=dim, dim_size=dim_size, reduce=self.reduce)
        return aggr

    def __repr__(self):
        return f"{self.__class__.__name__}(reduce={self.reduce})"


class SoftmaxAggregator(nn.Module):
    """ Softmax aggregator that inherits from nn.Module as the softmax aggregator is a parameterized module"""
    def __init__(self, phm_dim: int, initial_beta: float = 1.0, learn_beta: bool = True) -> None:
        super(SoftmaxAggregator, self).__init__()

        self.phm_dim = phm_dim
        self.initial_beta = initial_beta
        self.learn_beta = learn_beta
        self.beta = torch.nn.Parameter(torch.tensor(initial_beta), requires_grad=learn_beta)
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(self.initial_beta)

    def forward(self, x: torch.Tensor, idx: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> torch.Tensor:
        weights = torch_scatter.composite.scatter_softmax(src=self.beta * x, index=idx, dim=dim)
        x = weights * x
        x = torch_scatter.scatter(src=x, index=idx, dim=dim, dim_size=dim_size, reduce="sum")
        return x




## Code below is kindly taken from: https://github.com/lukecavabarrett/pna/tree/master/models/pytorch_geometric ##
# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

"""
    Base Aggregators
    Scatter-dimension is always 0, as the row-index corresponds to the node-index dimension.
    https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html
"""


def aggregate_sum(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src=src, index=index, dim=0, out=None, dim_size=dim_size, reduce='sum')


def aggregate_mean(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src=src, index=index, dim=0, out=None, dim_size=dim_size, reduce='mean')


def aggregate_min(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src=src, index=index, dim=0, out=None, dim_size=dim_size, reduce='min')


def aggregate_max(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src=src, index=index, dim=0, out=None, dim_size=dim_size, reduce='max')


def aggregate_var(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'var': aggregate_var,
    'std': aggregate_std,
}


""" 
   Base Scalers 
"""


def scale_identity(src: torch.Tensor, deg: torch.Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: torch.Tensor, deg: torch.Tensor, avg_deg: Dict[str, float]) -> torch.Tensor:
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: torch.Tensor, deg: torch.Tensor, avg_deg: Dict[str, float]) -> torch.Tensor:
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: torch.Tensor, deg: torch.Tensor, avg_deg: Dict[str, float]) -> torch.Tensor:
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: torch.Tensor, deg: torch.Tensor, avg_deg: Dict[str, float]) -> torch.Tensor:
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}


class PNAAggregator(nn.Module):
    """  Principal Neighbourhood Aggregator inherits from nn.Module in case we want to further parametrize. """

    def __init__(self, phm_dim: int, in_features: int, out_features: int, learn_phm: bool, init: str,
                 phm_rule, aggregators: List[str], scalers: Optional[List[str]],
                 deg: Optional[torch.Tensor]) -> None:
        super(PNAAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.aggregators_l = aggregators
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers_l = scalers
        if scalers:
            self.scalers = [SCALERS[scale] for scale in scalers]
            out_trafo_dim = in_features*(len(aggregators) * len(scalers))
            self.deg = deg.to(torch.float)
            self.avg_deg: Dict[str, float] = {
                'lin': self.deg.mean().item(),
                'log': (self.deg + 1).log().mean().item(),
                'exp': self.deg.exp().mean().item(),
            }
        else:
            self.scalers = None
            self.avg_deg = None
            out_trafo_dim = in_features*len(aggregators)


        self.transform = PHMLinear(in_features=out_trafo_dim, out_features=out_features, bias=True,
                                   phm_dim=phm_dim, phm_rule=phm_rule, learn_phm=learn_phm, init=init)

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()

    def forward(self, x: torch.Tensor, idx: torch.Tensor, dim_size: Optional[int] = None, dim: int = 0) -> torch.Tensor:
        outs = [aggr(x, idx, dim_size) for aggr in self.aggregators]
        # concatenate the different aggregator results, considering the shape of the hypercomplex components.
        out = phm_cat(tensors=outs, phm_dim=self.phm_dim, dim=-1)

        if self.scalers is not None:
            deg = degree(idx, dim_size, dtype=x.dtype).view(-1, 1)
            # concatenate the different aggregator results, considering the shape of the hypercomplex components.
            outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
            out = phm_cat(tensors=outs, phm_dim=self.phm_dim, dim=-1)

        out = self.transform(out)
        return out
