import torch
import torch.nn as nn
import torch_scatter

from typing import Optional, Dict

from phc.quaternion.algebra import QTensor


class Aggregator(nn.Module):
    """ Naive aggregator that inherits from nn.Module in case we want a parametrized version. """

    def __init__(self, reduce: str = "sum") -> None:
        super(Aggregator, self).__init__()
        assert reduce in ["sum", "min", "max", "mean"]
        self.reduce = reduce

    def reset_parameters(self):
        pass

    def forward(self, x: QTensor, idx: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> QTensor:
        x_tensor = x.stack(dim=1)  # transform to torch.Tensor  (*, 4, *)
        aggr = torch_scatter.scatter(src=x_tensor, index=idx, dim=dim, dim_size=dim_size, reduce=self.reduce)
        aggr = aggr.permute(1, 0, 2)  # permute such that first dimension is (4,*,*)
        return QTensor(*aggr)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduce={self.reduce})"


class SoftmaxAggregator(nn.Module):
    """ Softmax aggregator that inherits from nn.Module as the softmax aggregator is a parameterized module"""
    def __init__(self, initial_beta: float = 1.0, learn_beta: bool = True) -> None:
        super(SoftmaxAggregator, self).__init__()

        self.initial_beta = initial_beta
        self.learn_beta = learn_beta
        self.beta = torch.nn.Parameter(torch.tensor(initial_beta), requires_grad=learn_beta)
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(self.initial_beta)

    def forward(self, x: QTensor, idx: torch.Tensor, dim: int, dim_size: Optional[int] = None) -> QTensor:
        x = x.stack(dim=1)  # (num_nodes_batch, 4, feature_dim)
        weights = torch_scatter.composite.scatter_softmax(src=self.beta * x, index=idx, dim=dim)
        x = weights * x
        x = torch_scatter.scatter(src=x, index=idx, dim=dim, dim_size=dim_size, reduce="sum")
        # (num_nodes_batch, 4, feature_dim)
        x = x.permute(1, 0, 2)  # (4, num_nodes_batch, feature_dim)
        return QTensor(*x)


## Code below is kindly taken from: https://github.com/lukecavabarrett/pna/tree/master/models/pytorch_geometric ##
# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

"""
    Base Aggregators
"""


def aggregate_sum(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    return torch_scatter.scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src: torch.Tensor, index:torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src: torch.Tensor, index:torch.Tensor, dim_size: Optional[int]) -> torch.Tensor:
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