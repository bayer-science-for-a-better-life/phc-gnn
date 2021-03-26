import torch.nn.functional as F
import torch.nn as nn
import torch

from typing import Callable

from phc.quaternion.algebra import QTensor

r"""Swish activation function: """
sigmoid = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i) -> torch.Tensor:
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output) -> torch.Tensor:
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x) -> torch.Tensor:
        return swish(x)


swish_func = Swish_module()
""""""


r""" Quaternion split-activation functions"""


def qswish(q: QTensor) -> QTensor:
    return QTensor(swish_func(q.r), swish_func(q.i), swish_func(q.j), swish_func(q.k))


def qrelu(q: QTensor) -> QTensor:
    return QTensor(F.relu(q.r), F.relu(q.i), F.relu(q.j), F.relu(q.k))


def qrelu_naive(q: QTensor) -> QTensor:
    r"""
    quaternion relu activation function f(z), where z = a + b*i + c*j + d*k
    a,b,c,d are real scalars where the last three scalars correspond to the vectorial part of the quaternion number

    f(z) returns z, iif a + b + c + d > 0.
    Otherwise it returns 0
    Note that f(z) is applied for each dimensionality of q, as in real-valued fashion.

    :param q: quaternion tensor of shape (b, d) where b is the batch-size and d the dimensionality
    :return: activated quaternion tensor
    """
    q = q.stack(dim=0)
    sum = q.sum(dim=0)
    a = torch.heaviside(sum, values=torch.zeros(sum.size()).to(q.device))
    a = a.expand_as(q)
    q = q * a
    return QTensor(*q)


def qrelu_naive2(q: QTensor) -> QTensor:
    r"""
    quaternion relu activation function f(z), where z = a + b*i + c*j + d*k
    a,b,c,d are real scalars where the last three scalars correspond to the vectorial part of the quaternion number

    f(z) returns z, iif a,b,c,d > 0.
    Otherwise it returns 0
    Note that f(z) is applied for each dimensionality of q, as in real-valued fashion.

    :param q: quaternion tensor of shape (b, d) where b is the batch-size and d the dimensionality
    :return: activated quaternion tensor
    """
    mask_r, mask_i, mask_j, mask_k = q.r > 0.0,  q.i > 0.0,  q.j > 0.0,  q.k > 0.0
    mask = mask_r * mask_i * mask_j * mask_k
    r, i, j, k = mask * q.r, mask * q.i, mask * q.j, mask * q.k
    return QTensor(r, i, j, k)


def interaction(q: QTensor) -> torch.Tensor:
    norm = q.norm()
    c = norm.mean(dim=-1).unsqueeze(1)
    c = c.expand_as(norm)
    f = norm / torch.max(norm, c)
    return f


def qrelu_interaction(q: QTensor) -> QTensor:
    f = interaction(q)
    res = QTensor(f * q.r, f * q.i, f * q.j, f * q.k)
    return qrelu(res)


def qswish_interaction(q: QTensor) -> QTensor:
    f = interaction(q)
    res = QTensor(f * q.r, f * q.i, f * q.j, f * q.k)
    return qswish(res)


def qlrelu(q: QTensor) -> QTensor:
    return QTensor(F.leaky_relu(q.r), F.leaky_relu(q.i), F.leaky_relu(q.j), F.leaky_relu(q.k))


def qelu(q: QTensor) -> QTensor:
    return QTensor(F.elu(q.r), F.elu(q.i), F.elu(q.j), F.elu(q.k))


def qselu(q: QTensor) -> QTensor:
    return QTensor(F.selu(q.r), F.selu(q.i), F.selu(q.j), F.selu(q.k))


def get_functional_activation(activation: str) -> Callable:
    activation = activation.lower()
    if activation == "relu":
        return qrelu
    elif activation == "lrelu":
        return qlrelu
    elif activation == "elu":
        return qelu
    elif activation == "selu":
        return qselu
    elif activation == "swish":
        return qswish


def get_module_activation(activation: str) -> Callable:
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "lrelu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "swish":
        return swish_func
    elif activation == "identity":
        return nn.Identity()