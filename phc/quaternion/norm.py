import torch
import torch.nn as nn

from phc.quaternion.algebra import QTensor


r"""
Quaternion Batch Normalization:
This module is mainly rewritten from
https://github.com/gaudetcj/DeepQuaternionNetworks/blob/master/quaternion_layers/bn.py
to pytorch and usage of the QTensor class.

# also helpful: https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
Here in usage for standard (Quaternion) Fully-Connected-Layer where the Qtensor Class
holds the r,i,j,k tensors which each have shape [num_nodes_batch, d] where d is the feature dimension.
A stacked version of Qtensor into torch.tensor would be [num_nodes_batch, 4, d]
"""


def get_cholesky_decomposition_4x4(cov: torch.Tensor) -> torch.Tensor:
    r""" Manually computing the cholesky decomposition of a 4x4 matrix."""
    if cov.dim() == 3:
        # Cholesky decomposition of 4x4 symmetric matrix per feature
        Wrr = torch.sqrt(cov[:, 0, 0])

        Wri = (1.0 / Wrr) * (cov[:, 0, 1])
        Wii = torch.sqrt(cov[:, 1, 1] - (Wri * Wri))

        Wrj = (1.0 / Wrr) * (cov[:, 0, 2])
        Wij = (1.0 / Wii) * (cov[:, 1, 2] - (Wri * Wrj))
        Wjj = torch.sqrt((cov[:, 2, 2] - (Wij * Wij + Wrj * Wrj)))

        Wrk = (1.0 / Wrr) * (cov[:, 0, 3])
        Wik = (1.0 / Wii) * (cov[:, 1, 3] - (Wri * Wrk))
        Wjk = (1.0 / Wjj) * (cov[:, 2, 3] - (Wij * Wik + Wrj * Wrk))
        Wkk = torch.sqrt((cov[:, 3, 3] - (Wjk * Wjk + Wik * Wik + Wrk * Wrk)))
    elif cov.dim() == 4:
        Wrr = torch.sqrt(cov[:,:, 0, 0])

        Wri = (1.0 / Wrr) * (cov[:,:, 0, 1])
        Wii = torch.sqrt(cov[:,:, 1, 1] - (Wri * Wri))

        Wrj = (1.0 / Wrr) * (cov[:,:, 0, 2])
        Wij = (1.0 / Wii) * (cov[:,:, 1, 2] - (Wri * Wrj))
        Wjj = torch.sqrt((cov[:,:, 2, 2] - (Wij * Wij + Wrj * Wrj)))

        Wrk = (1.0 / Wrr) * (cov[:,:, 0, 3])
        Wik = (1.0 / Wii) * (cov[:,:, 1, 3] - (Wri * Wrk))
        Wjk = (1.0 / Wjj) * (cov[:,:, 2, 3] - (Wij * Wik + Wrj * Wrk))
        Wkk = torch.sqrt((cov[:,:, 3, 3] - (Wjk * Wjk + Wik * Wik + Wrk * Wrk)))


    sparse_l = {"rr": Wrr,
                "ri": Wri, "ii": Wii,
                "rj": Wrj, "ij": Wij, "jj": Wjj,
                "rk": Wrk, "ik": Wik, "jk": Wjk, "kk": Wkk}

    l = get_lower_triangular_from_sparsedict(sparse_l)

    return l


def get_lower_triangular_from_sparsedict(sparse_l: dict):
    r""" returns the cholesky decomposition from a sparse dictionary to full matrix"""
    p = sparse_l["rr"].size(-1)
    dim_in = sparse_l["rr"].dim()
    shape = tuple(sparse_l["rr"].size()) + (4, 4)
    l = torch.zeros(shape)
    if dim_in == 1:
        for t in range(p):
            l[t, 0, 0] = sparse_l.get("rr")[t]

            l[t, 1, 0] = sparse_l.get("ri")[t]
            l[t, 1, 1] = sparse_l.get("ii")[t]

            l[t, 2, 0] = sparse_l.get("rj")[t]
            l[t, 2, 1] = sparse_l.get("ij")[t]
            l[t, 2, 2] = sparse_l.get("jj")[t]

            l[t, 3, 0] = sparse_l.get("rk")[t]
            l[t, 3, 1] = sparse_l.get("ik")[t]
            l[t, 3, 2] = sparse_l.get("jk")[t]
            l[t, 3, 3] = sparse_l.get("kk")[t]
    elif dim_in == 2:
        for b in range(shape[0]):
            for t in range(p):
                l[b, t, 0, 0] = sparse_l.get("rr")[b][t]

                l[b, t, 1, 0] = sparse_l.get("ri")[b][t]
                l[b, t, 1, 1] = sparse_l.get("ii")[b][t]

                l[b, t, 2, 0] = sparse_l.get("rj")[b][t]
                l[b, t, 2, 1] = sparse_l.get("ij")[b][t]
                l[b, t, 2, 2] = sparse_l.get("jj")[b][t]

                l[b, t, 3, 0] = sparse_l.get("rk")[b][t]
                l[b, t, 3, 1] = sparse_l.get("ik")[b][t]
                l[b, t, 3, 2] = sparse_l.get("jk")[b][t]
                l[b, t, 3, 3] = sparse_l.get("kk")[b][t]

    return l


def whiten4x4(q: torch.Tensor, training=True, running_mean=None, running_cov=None, momentum=0.1, nugget=1e-5):
    r"""
    Whitens a data matrix to have zero mean and unit covariance following the equation:
    x_whitened = W @ E[x - x_mean]
    W turns out to be the inverse of a lower-triangular cholesky matrix and x-x_mean is the centered data.
    """
    # q has size [4, batch_size, p]
    if training:
        qmean = q.mean(dim=1)  # [4, p]
        if running_mean is not None:
            with torch.no_grad():
                running_mean += momentum * (qmean.data - running_mean)
    else:
        qmean = running_mean

    # get centered version broadcasting
    qmean = qmean.unsqueeze(dim=1)  # [4, 1, p]
    qcentered = q - qmean  # [4, batch_size, p]
    if training:
        perm = qcentered.permute(2, 0, 1)  # [p, 4, batch_size]
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]  # [p, 4, 4]
        if running_cov is not None:
            with torch.no_grad():
                running_cov += momentum * (cov.data.permute(1, 2, 0) - running_cov)
    else:
        cov = running_cov.permute(2, 0, 1)

    cov = cov.to(q.device)
    eye = nugget * torch.eye(4, device=q.device, dtype=cov.dtype).unsqueeze(0)
    eye = eye.to(q.device)
    v = (cov + eye).to(q.device)
    #l = get_cholesky_decomposition_4x4(cov=v)
    l = torch.cholesky(input=v, upper=False)   # lets see if this is faster and runs without errors...
    l = l.to(q.device)  # [p, 4, 4]
    l_inv = l.inverse()
    #l_inv = torch.cholesky_inverse(input=l, upper=False)   # throws error...

    rhs = qcentered.unsqueeze(-1).permute(*range(1, qcentered.dim()), 0, -1)  # [batch_size, p, 4, 1]
    # add broadcasting dimension
    l_inv = l_inv.unsqueeze(0)
    xx = l_inv.matmul(rhs)
    xx = xx.squeeze(-1)
    res = torch.stack(torch.unbind(xx, dim=-1), dim=0)

    # assert
    do_assert = False

    if do_assert:
        perm = res.permute(2, 0, 1)  # [p, 4, batch_size]
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]  # [p, 4, 4]
        sums = []
        for c in cov:
            sums.append(torch.sum(torch.diag(c)) / 4.0)
        assert abs(1 - torch.mean(torch.tensor(sums)).item()) < 1e-4
        assert abs(res.mean(dim=1).norm().item()) < 1e-4

    return res


def quaternion_batch_norm(
    qtensor: QTensor,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=True,
    momentum=0.1,
    eps=1e-05,
) -> QTensor:
    """ Functional implementation of quaternion batch normalization """
    # check arguments
    assert ((running_mean is None and running_var is None)
            or (running_mean is not None and running_var is not None))
    assert ((weight is None and bias is None)
            or (weight is not None and bias is not None))

    # stack qtensor along the first dimension
    x = qtensor.stack(dim=0)
    # whiten and apply affine transformation
    z = whiten4x4(q=x, training=training, running_mean=running_mean,
                  running_cov=running_var, momentum=momentum, nugget=eps)

    p = x.size(-1)
    if weight is not None and bias is not None:
        shape = (1, p)
        weight = weight.reshape(4, 4, *shape)

        """ this is just the scaling formula
         x_r_BN = gamma_rr * x_r + gamma_ri * x_i + gamma_rj * x_j + gamma_rk * x_k + beta_r
         x_i_BN = gamma_ir * x_r + gamma_ii * x_i + gamma_ij * x_j + gamma_ik * x_k + beta_i
         x_j_BN = gamma_jr * x_r + gamma_ji * x_i + gamma_jj * x_j + gamma_jk * x_k + beta_j
         x_k_BN = gamma_kr * x_r + gamma_ki * x_i + gamma_kj * x_j + gamma_kk * x_k + beta_k
        """

        z = torch.stack([
            z[0] * weight[0, 0] + z[1] * weight[0, 1] + z[2] * weight[0, 2] + z[3] * weight[0, 3],
            z[0] * weight[1, 0] + z[1] * weight[1, 1] + z[2] * weight[1, 2] + z[3] * weight[1, 3],
            z[0] * weight[2, 0] + z[1] * weight[2, 1] + z[2] * weight[2, 2] + z[3] * weight[2, 3],
            z[0] * weight[3, 0] + z[1] * weight[3, 1] + z[2] * weight[3, 2] + z[3] * weight[3, 3],
        ], dim=0) + bias.reshape(4, *shape)

    return QTensor(z[0], z[1], z[2], z[3])


class QuaternionBatchNorm(torch.nn.Module):
    r""" torch.nn.Module for the quaternion batch-normalization. """
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5,
                 affine: bool = True, track_running_stats: bool = True) -> None:

        super(QuaternionBatchNorm, self).__init__()
        self.affine = affine
        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(4, 4, num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor(4, num_features), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.empty(4, num_features))
            self.register_buffer('running_var', torch.empty(4, 4, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.bias.data.fill_(0.0)
            self.weight.data.fill_(0.0)
            self.weight.data[0, 0].fill_(0.5)
            self.weight.data[1, 1].fill_(0.5)
            self.weight.data[2, 2].fill_(0.5)
            self.weight.data[3, 3].fill_(0.5)

    def forward(self, q: QTensor, **kwargs) -> QTensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        q = quaternion_batch_norm(q, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training or not self.track_running_stats,
                                  exponential_average_factor, self.eps)

        return q

    def extra_repr(self) -> str:
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)



class NaiveQuaternionBatchNorm(torch.nn.Module):
    """ Implements the naive quaternion batch-normalization, where the batch-norm is applied separately for
        each component of the quaternion number, i.e. r, i, j, k.
    """
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        
        super(NaiveQuaternionBatchNorm, self).__init__()

        self.bn = torch.nn.ModuleDict({"r": nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats),
                                       "i": nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats),
                                       "j": nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats),
                                       "k": nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
                                       })
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.bn.values():
            module.reset_parameters()

    def forward(self, x: QTensor) -> QTensor:
        r, i, j, k = self.bn["r"](x.r), self.bn["i"](x.i), self.bn["j"](x.j), self.bn["k"](x.k)
        return QTensor(r, i, j, k)

    def __repr__(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats})'.format(**self.__dict__)





""" Wrapper for quaternion batch-normalization """


class QuaternionNorm(torch.nn.Module):
    r"""
    Implements the normalization layer for quaternion-valued tensors.
    Can use type "naive-batch-norm" and "q-batch-norm".
    """
    def __init__(self, num_features: int, type: str = "naive-batch-norm", **kwargs):
        super(QuaternionNorm, self).__init__()
        assert type in ["naive-batch-norm", "q-batch-norm"]
        self.type = type
        self.num_features = num_features
        self.kwargs = kwargs
        if type == "q-batch-norm":
            self.bn = QuaternionBatchNorm(num_features=num_features, **kwargs)
        else:
            self.bn = NaiveQuaternionBatchNorm(num_features=num_features, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.bn.reset_parameters()


    def forward(self, x: QTensor) -> QTensor:
        return self.bn(x)


    def __repr__(self):
        return f'{self.__class__.__name__}:(num_features={self.num_features}, type={self.type})'