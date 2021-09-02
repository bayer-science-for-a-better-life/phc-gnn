import torch
import torch.nn as nn


class NaivePHMNorm(torch.nn.Module):
    """ Implements the naive hypercomplex batch-normalization, where the batch-norm is applied separately for
        each component of the hypercomplex number.
    """

    def __init__(self, num_features: int, phm_dim: int, momentum: float = 0.1, eps: float = 1e-5,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(NaivePHMNorm, self).__init__()

        self.phm_dim = phm_dim
        self.num_features = num_features // phm_dim
        self.momentum = momentum
        self.affine = affine
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
                                for _ in range(phm_dim)])

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.bn:
            module.reset_parameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, self.phm_dim * self.num_features)
        x = x.reshape(x.size(0), self.phm_dim, self.num_features).permute(1, 0, 2)
        x = [bn_i(x_i) for bn_i, x_i in zip(self.bn, x)]
        x = torch.cat(x, dim=-1)
        return x

    def __repr__(self):
        return '{num_features}, phm_dim={phm_dim}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats})'.format(**self.__dict__)


""" Wrapper for PHM tensor batch-normalization """


class PHMNorm(torch.nn.Module):
    r"""
    Implements the normalization layer for hypercomplex-valued tensors.
    Can use type "naive-batch-norm" and "naive-naive-batch-norm" only.
    """

    def __init__(self, num_features: int, phm_dim: int, type: str = "naive-batch-norm", **kwargs):
        super(PHMNorm, self).__init__()
        assert type in ["naive-batch-norm", "naive-naive-batch-norm"]
        self.type = type
        self.num_features = num_features
        self.phm_dim = phm_dim
        self.kwargs = kwargs
        if type == "naive-batch-norm":
            self.bn = NaivePHMNorm(num_features=num_features, phm_dim=phm_dim, **kwargs)
        elif type == "naive-naive-batch-norm":
            self.bn = torch.nn.BatchNorm1d(num_features=num_features*phm_dim, **kwargs)
        else:
            raise ValueError

        self.reset_parameters()

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)

    def __repr__(self):
        return f'{self.__class__.__name__}:(num_features={self.num_features}, phm_dim={self.phm_dim} type={self.type})'