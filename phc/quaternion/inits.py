import numpy as np
import torch
from numpy.random import RandomState
from scipy.stats import chi
from torch import Tensor
import math

from phc.quaternion.algebra import QTensor
from phc.quaternion.qr import quat_QR, RealP


def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))


def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))


def glorot_orthogonal(tensor: torch.Tensor, scale: float = 2.0):
    r""" Taken from: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/inits.py"""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def unitary_init(in_features, out_features, low=0, high=1, dtype=torch.float64) -> (Tensor, Tensor, Tensor, Tensor):
    # init in interval [low, high], i.e. with defaults: 0 and 1
    v_r = torch.FloatTensor(in_features, out_features).to(dtype).zero_()
    v_i = torch.FloatTensor(in_features, out_features).to(dtype).uniform_(low, high)
    v_j = torch.FloatTensor(in_features, out_features).to(dtype).uniform_(low, high)
    v_k = torch.FloatTensor(in_features, out_features).to(dtype).uniform_(low, high)
    # Unitary quaternion
    q = QTensor(v_r, v_i, v_j, v_k)
    q_unitary = q.normalize()
    return q_unitary.r, q_unitary.i, q_unitary.j, q_unitary.k


def quaternion_init(in_features: int, out_features: int, criterion: str = 'glorot',
                    low: int = 0, high: int = 1, transpose: bool = True) -> (Tensor, Tensor, Tensor, Tensor):

    fan_in = in_features
    fan_out = out_features
    if criterion == 'glorot':
        s = 1. / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)


    kernel_shape = (in_features, out_features)
    magnitude = torch.from_numpy(chi.rvs(df=4, loc=0, scale=s, size=kernel_shape)).to(torch.float64)
    # Purely imaginary quaternions unitary
    _, v_i, v_j, v_k = unitary_init(in_features, out_features, low, high)

    theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=kernel_shape)).to(torch.float64)
    phi_i = torch.cos(torch.from_numpy(np.random.uniform(low=-s, high=s, size=kernel_shape)).to(torch.float64))**2
    phi_j = torch.cos(torch.from_numpy(np.random.uniform(low=-s, high=s, size=kernel_shape)).to(torch.float64))**2
    phi_k = torch.cos(torch.from_numpy(np.random.uniform(low=-s, high=s, size=kernel_shape)).to(torch.float64))**2
    phi = phi_i / (phi_i + phi_j + phi_k)
    phj = phi_j / (phi_i + phi_j + phi_k)
    phk = phi_k / (phi_i + phi_j + phi_k)
    weight_r = magnitude * torch.cos(theta)
    weight_i = magnitude * v_i * torch.sin(theta) * phi
    weight_j = magnitude * v_j * torch.sin(theta) * phj
    weight_k = magnitude * v_k * torch.sin(theta) * phk

    if transpose:
        weight_r = weight_r.t()
        weight_i = weight_i.t()
        weight_j = weight_j.t()
        weight_k = weight_k.t()

    return weight_r.to(torch.float32), weight_i.to(torch.float32), weight_j.to(torch.float32), weight_k.to(torch.float32)


def orthogonal_init(in_features: int, out_features: int,
                    scale: float = 1.0, transpose: bool = True) -> (Tensor, Tensor, Tensor, Tensor):

    W = torch.zeros(4, in_features, out_features, dtype=torch.float64).normal_(std=scale)
    #W = torch.zeros(4, in_features, out_features, dtype=torch.float32).normal_(std=scale)

    if transpose:
        W = W.permute(0, 2, 1)

    m, n = W.size(1), W.size(2)
    if m < n:
        W = W.permute(0, 2, 1)

    Q, R = quat_QR(A1=W[0], A2=W[1], A3=W[2], A4=W[3])

    Q /= 2.0
    # cast back to torch.float32
    Q = Q.to(torch.float32)


    # small assert
    # W_unitary = RealP(Wr, Wi, Wj, Wk)
    # diff = abs(torch.diag(W_unitary @ W_unitary.T) - torch.ones(4*out_features))
    # print(diff.sum())


    Q = Q[:, :in_features]
    if m < n:
        Wr, Wi, Wj, Wk = Q.split(in_features, dim=0)
        Wr, Wi, Wj, Wk = Wr[:out_features, :], Wi[:out_features, :], Wj[:out_features, :], Wk[:out_features, :]
    else:
        Wr, Wi, Wj, Wk = Q.split(out_features, dim=0)

    return Wr, Wi, Wj, Wk

