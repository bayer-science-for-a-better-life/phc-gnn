import torch
import numpy as np
from typing import Union
import warnings

OptTensor = Union[torch.Tensor, float, None]
QTensorable = Union["QTensor", torch.Tensor, float, None]


def ensure_qtensor(tensorable: QTensorable) -> "QTensor":
    """Ensures that the input data is converted to a Quaternion Tensor"""
    if isinstance(tensorable, QTensor) or tensorable.__class__.__name__ == "QTensor":
        return tensorable
    elif isinstance(tensorable, (torch.Tensor, float)):
        return QTensor(r=tensorable, i=None, j=None, k=None)
    else:
        warnings.warn(f"Data type {type(tensorable)} was inserted.")
        #print(f"Data type {type(tensorable)} was inserted.")
        raise ValueError


r"""
Major kudos to: https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/cplx.py where the main class was adapted
to match the Quaternion tensors.
"""


class QTensor(object):
    r"""A type partially implementing quaternion valued tensors in torch.
    Creates a quaternion tensor object from the real and three hyper-quaternion torch tensors or pythonic floats.
    """
    __slots__ = ("__r", "__i", "__j", "__k", "__is_real")

    def __init__(self, r: Union[float, torch.Tensor],
                 i: OptTensor = None, j: OptTensor = None, k: OptTensor = None):
        #is_real = False
        if isinstance(r, float):
            if i is None:
                #is_real = True
                i = 0.0
            if j is None:
                #is_real = True
                j = 0.0
            if k is None:
                #is_real = True
                k = 0.0
            elif not (isinstance(i, float) and isinstance(j, float) and isinstance(k, float)):
                raise TypeError("""quaternion parts must be float.""")

            r, i, j, k = torch.tensor(r), torch.tensor(i), torch.tensor(j), torch.tensor(k)

        if i is None:
            #is_real = True
            i = torch.zeros_like(r)

        if j is None:
            #is_real = True
            j = torch.zeros_like(r)

        if k is None:
            #is_real = True
            k = torch.zeros_like(r)

        if r.shape != i.shape != j.shape != k.shape:
            raise ValueError("""Real and imaginary parts have """
                             """mismatching shape.""")

        # set the properties
        self.__r, self.__i, self.__j, self.__k = r, i, j, k

        # check if it's a real quaternion
        #self.__is_real = is_real
        self.__is_real = _is_real_quaternion(self)


    @property
    def r(self):
        r"""Real part of the quaternion tensor."""
        return self.__r

    @property
    def i(self):
        r"""Imaginary i part of the quaternion tensor."""
        return self.__i

    @property
    def j(self):
        r"""Imaginary j part of the quaternion tensor."""
        return self.__j

    @property
    def k(self):
        r"""Imaginary k part of the quaternion tensor."""
        return self.__k

    @property
    def is_real(self):
        return self.__is_real

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" \
               f" r={self.__r},\n" \
               f" i={self.__i},\n" \
               f" j={self.__j},\n" \
               f" k={self.__k}\n" \
               f")"

    def __str__(self):
        to_print = f"r: {self.__r}, \n" \
                   f"i: {self.__i}, \n" \
                   f"j: {self.__j}, \n" \
                   f"k: {self.__k}, \n"
        return to_print

    @staticmethod
    def check_type(p):
        return p.__class__.__name__ == "QTensor"

    def __getitem__(self, key):
        r"""Index the quaternion tensor."""
        return type(self)(self.__r[key], self.__i[key], self.__j[key], self.__k[key])

    def __setitem__(self, key, value):
        if not isinstance(value, QTensor):
            self.__r[key], self.__i[key], self.__j[key], self.__k[key] = value, value, value, value
        else:
            self.__r[key], self.__i[key], self.__r[key], self.__k[key] = value.r, value.i, value.j, value.k

    def clone(self):
        r"""Clone a quaternion tensor."""
        return type(self)(self.__r.clone(), self.__i.clone(), self.__j.clone(), self.__k.clone())

    def get(self, part):
        if part == "r":
            return self.r
        elif part == "i":
            return self.i
        elif part == "j":
            return self.j
        elif part == "k":
            return self.k
        else:
            raise ValueError

    def to_real_matrix(self):
        return get_real_matrix_representation(Q=self)

    def __pos__(self):
        r"""Return the quaternion tensor as is."""
        return _pos(self)

    def __neg__(self):
        r"""Flip the sign of the quaternion tensor."""
        return _neg(self)

    def __add__(self, other):
        r"""Addition of quaternion tensors."""
        return _add(self, ensure_qtensor(other))

    def __radd__(self, other):
        r""" Right addition when other + self"""
        return _add(ensure_qtensor(other), self)

    def __iadd__(self, other):
        r"""In place addition, when self += other. Note, returns new object"""
        return _add(self, ensure_qtensor(other))

    def __sub__(self, other):
        r"""Subtraction of quaternion tensors."""
        return _sub(self, ensure_qtensor(other))

    def __rsub__(self, other):
        r"""Subtraction of quaternion tensors."""
        return _sub(ensure_qtensor(other), self)

    def __isub__(self, other):
        r"""In place subtraction. Note, creates new object."""
        return _sub(self, ensure_qtensor(other))

    def __mul__(self, other):
        r"""Elementwise product of quaternion tensors.
        Hamilton product reference: https://de.mathworks.com/help/aerotbx/ug/quatmultiply.html

        computes self * other"""
        return _mul(self, ensure_qtensor(other))

    def __rmul__(self, other):
        r""" computes other * self """
        return _rmul(self, ensure_qtensor(other))

    def __imul__(self, other):
        return _mul(self, ensure_qtensor(other))

    def __matmul__(self, other):
        return _matmul(self, ensure_qtensor(other))

    def __rmatmul__(self, other):
        raise NotImplementedError

    def __pow__(self, power):
        r"""Elementwise Power for Quaternions. Note that this is not the function f(q) = q^power"""
        assert type(power) == float or type(power) == int, f"power must be of type int or float. Type {type(power)}" \
                                                           f" was inserted."
        return type(self)(self.__r ** power, self.__i ** power, self.__j ** power, self.__k ** power)

    def __truediv__(self, other):
        r"""
        Elementwise division of quaternion tensors.
        Reference: https://de.mathworks.com/help/aerotbx/ug/quatdivide.html
        """
        return _div(self, ensure_qtensor(other))

    def __rtruediv__(self, other):
        r"""
        Elementwise division of quaternion tensors.
        Reference: https://de.mathworks.com/help/aerotbx/ug/quatdivide.html
        """
        return _div(ensure_qtensor(other), self)

    def __eq__(q, p):
        if q.check_type(p):  # check for real type
            return torch.allclose(q.__r, p.r) and torch.allclose(q.__i, p.i) \
                   and torch.allclose(q.__j, p.j) and torch.allclose(q.__k, p.k)
        else:
            return torch.allclose(q.__r, type(q)(p).r)

    @property
    def conj(self):
        r"""The conjugate of the quaternion tensor."""
        return _conjugate(self)

    def conjugate(self):
        r"""The quaternion conjugate of the quaternion tensor."""
        return self.conj

    @property
    def mod(self):
        r"""The quaternion modulus, aka. "norm" of a quaternion representation if seen as a 4-d vector
        Reference: https://de.mathworks.com/help/aerotbx/ug/quatmod.html"""
        return _modulus(self)

    def modulus(self):
        r"""The quaternion modulus, aka. "norm" of a quaternion representation if seen as a 4-d vector
        Reference: https://de.mathworks.com/help/aerotbx/ug/quatmod.html"""
        return self.mod

    def __abs__(self):
        return self.modulus()

    def norm(self):
        r"""The norm of a quaternion. Reference: https://de.mathworks.com/help/aerotbx/ug/quatnorm.html
         Changed to also include the square-root as described in
         https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal"""
        return self.mod

    def inverse(self):
        r"""The inverse of a quaternion. Reference: https://de.mathworks.com/help/aerotbx/ug/quatinv.html"""
        return _inverse(self)

    def normalize(self, eps: float = 1E-10):
        r"""Normalize a quaterion representation.
        Reference: https://de.mathworks.com/help/aerotbx/ug/quatnormalize.html"""
        return _normalize(self, eps)

    def dot(self, other):
        r"""Computes the dot/inner product between two quaternions."""
        return _dot(self, ensure_qtensor(other))

    def stack(q, dim=0) -> torch.Tensor:
        return torch.stack(tensors=[q.__r, q.__i, q.__j, q.__k], dim=dim)

    def mean(q, dim=0):
        r_mean = q.__r.mean(dim=dim)
        i_mean = q.__i.mean(dim=dim)
        j_mean = q.__j.mean(dim=dim)
        k_mean = q.__k.mean(dim=dim)
        return type(q)(r_mean, i_mean, j_mean, k_mean)

    @property
    def shape(self):
        r"""Returns the shape of the quaternion tensor."""
        return self.__r.shape

    def __len__(self):
        r"""The size of the zero-th dimension of the quaternion tensor."""
        return self.shape[0]

    def t(self):
        r"""The transpose of a 2d quaternion tensor."""
        return type(self)(self.__r.t(), self.__i.t(),  self.__j.t(),  self.__k.t())

    @property
    def T(self):
        r"""The transpose of a 2d quaternion tensor."""
        return self.t()

    def view(self, *shape):
        r"""Return a view of the quaternion tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return type(self)(self.__r.view(*shape), self.__i.view(*shape), self.__j.view(*shape), self.__k.view(*shape))

    def view_as(self, other):
        r"""Return a view of the quaternion tensor of shape other."""
        shape = other.shape
        return self.view(*shape)

    def reshape(self, *shape):
        r"""Reshape the quaternion tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return type(self)(self.__r.reshape(*shape), self.__i.reshape(*shape),
                          self.__j.reshape(*shape), self.__k.reshape(*shape))

    def size(self, *dim):
        r"""Returns the size of the quaternion tensor."""
        return self.__r.size(*dim)

    def squeeze(self, dim=None):
        r"""Returns the quaternion tensor with all the dimensions of input of size 1 removed."""
        return _squeeze(self, dim)

    def unsqueeze(self, dim=None):
        r"""Returns a new quaternion tensor with a dimension of size one inserted at the specified position."""
        return _unsqueeze(self, dim)

    def detach(self):
        r"""Return a copy of the quaternion tensor detached from autograd graph."""
        return type(self)(self.__r.detach(), self.__i.detach(), self.__j.detach(), self.__k.detach())

    def requires_grad_(self, requires_grad=True):
        # By default the requires_grad is set to False in the constructor.
        r"""Toggle the gradient of real and hypercomplex parts."""
        return type(self)(self.__r.requires_grad_(requires_grad),
                          self.__i.requires_grad_(requires_grad),
                          self.__j.requires_grad_(requires_grad),
                          self.__k.requires_grad_(requires_grad))

    @property
    def grad(self):
        r"""Collect the accumulated gradinet of the quaternion tensor."""
        r, i, j, k = self.__r.grad, self.__i.grad,  self.__j.grad,  self.__k.grad
        return None if r is None or i is None or j is None or k is None else type(self)(r, i, j, k)

    def cuda(self, device=None, non_blocking=False):
        r"""Move the quaternion tensor to a CUDA device."""
        r = self.__r.cuda(device=device, non_blocking=non_blocking)
        i = self.__i.cuda(device=device, non_blocking=non_blocking)
        j = self.__j.cuda(device=device, non_blocking=non_blocking)
        k = self.__k.cuda(device=device, non_blocking=non_blocking)
        return type(self)(r, i, j, k)

    def cpu(self):
        r"""Move the quaternion tensor to CPU."""
        return type(self)(self.__r.cpu(), self.__i.cpu(), self.__j.cpu(), self.__k.cpu())

    def to(self, *args, **kwargs):
        r"""Move / typecast the quaternion tensor."""
        return type(self)(self.__r.to(*args, **kwargs),
                          self.__i.to(*args, **kwargs),
                          self.__j.to(*args, **kwargs),
                          self.__k.to(*args, **kwargs))

    @property
    def device(self):
        r"""The hosting device of the quaternion tensor."""
        return self.__r.device

    @property
    def dtype(self):
        r"""The base dtype of the quaternion tensor."""
        return self.__r.dtype

    @property
    def requires_grad(self):
        return self.r.requires_grad

    def backward(self, grad=None):
        if grad is None:
            grad = torch.ones_like(self.__r)

        self.__r.backward(grad)
        self.__i.backward(grad)
        self.__j.backward(grad)
        self.__k.backward(grad)

    def dim(self):
        r"""The number of dimensions in the quaternion tensor."""
        return len(self.shape)

    def permute(self, *dims):
        r"""Shuffle the dimensions of the quaternion tensor."""
        return _permute(self, *dims)

    def transpose(self, dim0, dim1):
        r"""Transpose the specified dimensions of the quaternion tensor."""
        return _tranpose(self, dim0, dim1)

    @classmethod
    def empty(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create an empty quaternion tensor."""
        r = torch.empty(*sizes, dtype=dtype, device=device,
                        requires_grad=requires_grad)
        return cls(r, torch.empty_like(r, requires_grad=requires_grad),
                   torch.empty_like(r, requires_grad=requires_grad),
                   torch.empty_like(r, requires_grad=requires_grad))

    @classmethod
    def zeros(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create an empty quaternion tensor."""
        r = torch.zeros(*sizes, dtype=dtype, device=device,
                         requires_grad=requires_grad)
        return cls(r, torch.zeros_like(r, requires_grad=requires_grad),
                   torch.zeros_like(r, requires_grad=requires_grad),
                   torch.zeros_like(r, requires_grad=requires_grad))

    @classmethod
    def ones(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create a ones quaternion tensor."""
        r = torch.ones(*sizes, dtype=dtype, device=device,
                        requires_grad=requires_grad)
        return cls(r, torch.ones_like(r, requires_grad=requires_grad),
                   torch.ones_like(r, requires_grad=requires_grad),
                   torch.ones_like(r, requires_grad=requires_grad))

    @classmethod
    def eye(self, *sizes, dtype=None, device=None, requires_grad=False):
        r = torch.eye(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        i = torch.eye(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        j = torch.eye(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.eye(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        return self(r, i, j, k)



r""" 
Implementation of Primitives
- as of now we rely on pytorch's built-in auto-grad function for the separate components. -
"""


def _is_real_quaternion(q: QTensor) -> bool:
    is_real = torch.allclose(q.i, torch.zeros_like(q.i).to(q.device)) and \
              torch.allclose(q.j, torch.zeros_like(q.j).to(q.device)) and \
              torch.allclose(q.k, torch.zeros_like(q.k).to(q.device))
    return is_real


def _pos(q: QTensor) -> QTensor:
    return QTensor(q.r, q.i, q.j, q.k)


def _conjugate(q: QTensor) -> QTensor:
    r"""Returns the conjugate representation of a quaternion number.
        E.g. q = a + b*i + c*j + d*k,  then the conjugate representation is
        q_conjugate = a - b*i - c*j - d*k"""

    return QTensor(q.r, -q.i, -q.j, -q.k)


def _dot(q1: QTensor, q2: QTensor) -> torch.Tensor:
    r"""Returns the dot product between quaternion numbers"""
    return q1.r * q2.r + q1.i * q2.i + q1.j * q2.j + q1.k * q2.k


def _modulus(q: QTensor) -> torch.Tensor:
    r"""Returns the modulus / norm of a quaternion number"""
    return torch.sqrt(_dot(q, q))


def _add(q1: QTensor, q2: QTensor) -> QTensor:
    r""" Addition: q1 + q2 """
    r = q1.r + q2.r
    if not q2.is_real:  # include is_real check to remove redundant computations and stability
        i = q1.i + q2.i
        j = q1.j + q2.j
        k = q1.k + q2.k
    else:
        i, j, k = q1.i, q1.j, q1.k

    return QTensor(r, i, j, k)


def _mul(q1: QTensor, q2: QTensor) -> QTensor:
    r"""Quaternion multiplication / Hamilton product:  q1 * q2 """
    if not q2.is_real:
        r = q1.r * q2.r - q1.i * q2.i - q1.j * q2.j - q1.k * q2.k
        i = q1.i * q2.r + q1.r * q2.i - q1.k * q2.j + q1.j * q2.k
        j = q1.j * q2.r + q1.k * q2.i + q1.r * q2.j - q1.i * q2.k
        k = q1.k * q2.r - q1.j * q2.i + q1.i * q2.j + q1.r * q2.k
    else:  # hadamard product with q1 and the real part of q2  to reduce redundant zero-multiplication
        r = q1.r * q2.r
        i = q1.i * q2.r
        j = q1.j * q2.r
        k = q1.k * q2.r
    return QTensor(r, i, j, k)


def _rmul(q1: QTensor, q2: QTensor) -> QTensor:
    r""" Quaternion multiplication q2 * q1"""
    if q2.is_real:
        r = q1.r * q2.r
        i = q1.i * q2.r
        j = q1.j * q2.r
        k = q1.k * q2.r
        return QTensor(r, i, j, k)
    else:
        return _mul(q2, q1)


def _inverse(q: QTensor) -> QTensor:
    r"""Returns the inverse quaternion"""
    q_conj = q.conjugate()
    q_norm = q.norm() ** 2
    r = q_conj.r / q_norm
    if not q.is_real:
        i = q_conj.i / q_norm
        j = q_conj.j / q_norm
        k = q_conj.k / q_norm
    else:
        i, j, k = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
    q_inv = QTensor(r, i, j, k)
    return q_inv


def _div(q1: QTensor, q2: QTensor) -> QTensor:
    r"""Quaternion division q1 / q2 .
        The division equals to q1 * q2_inv, where q2_inv is the inverse of q2."""
    q2_inv = q2.inverse()
    return _mul(q1, q2_inv)


def _normalize(q: QTensor, eps: float = 1e-10) -> QTensor:
    norm = q.norm() + eps
    q_normalized = QTensor(q.r / norm, q.i / norm, q.j / norm, q.k / norm)
    return q_normalized


def _neg(q: QTensor) -> QTensor:
    return QTensor(-q.r, -q.i, -q.j, -q.k)


def _sub(q1: QTensor, q2: QTensor) -> QTensor:
    r = q1.r - q2.r
    i = q1.i - q2.i
    j = q1.j - q2.j
    k = q1.k - q2.k
    return QTensor(r, i, j, k)



def _squeeze(q: QTensor, dim=None) -> QTensor:
    if dim is None:
        return QTensor(q.r.squeeze(), q.i.squeeze(), q.j.squeeze(), q.k.squeeze())
    else:
        return QTensor(q.r.squeeze(dim=dim), q.i.squeeze(dim=dim), q.j.squeeze(dim=dim), q.k.squeeze(dim=dim))


def _unsqueeze(q: QTensor, dim=None) -> QTensor:
    if dim is None:
        return QTensor(q.r.unsqueeze(), q.i.unsqueeze(), q.j.unsqueeze(), q.k.unsqueeze())
    else:
        return QTensor(q.r.unsqueeze(dim=dim), q.i.unsqueeze(dim=dim), q.j.unsqueeze(dim=dim), q.k.unsqueeze(dim=dim))


def _permute(q: QTensor, *dims) -> QTensor:
    return QTensor(q.r.permute(*dims), q.i.permute(*dims), q.j.permute(*dims), q.k.permute(*dims))


def _tranpose(q: QTensor, dim0: int, dim1: int) -> QTensor:
    return QTensor(q.r.transpose(dim0, dim1), q.i.transpose(dim0, dim1),
                   q.j.transpose(dim0, dim1), q.k.transpose(dim0, dim1))


def _matmul(q1: QTensor, q2: QTensor) -> QTensor:
    r"""
    q1 has shape (out_channels, in_channels)
    q2 has shape (in_channels, *)
    where * is either empty or an integer
    """
    row_dims = q1.size(0)
    assert q1.size(1) == q2.size(0)
    # get real matrix representation of q1
    real_q1 = get_real_matrix_representation(q1)
    # stack q2 row-wise
    stacked_q2 = torch.cat([q2.r, q2.i, q2.j, q2.k], dim=0)
    # apply matrix-matrix/vector multiplication
    out = torch.matmul(real_q1, stacked_q2)
    splits = out.split(split_size=row_dims, dim=0)
    out = QTensor(*splits)
    return out


def cat(tensors: Union[list, tuple], dim: int) -> QTensor:
    r = torch.cat([z.r for z in tensors], dim=dim)
    i = torch.cat([z.i for z in tensors], dim=dim)
    j = torch.cat([z.j for z in tensors], dim=dim)
    k = torch.cat([z.k for z in tensors], dim=dim)
    return QTensor(r, i, j, k)


def get_real_matrix_representation(Q: QTensor) -> torch.Tensor:
    r"""
    Returns one of the the real matrix representation of a quaternion tensor
    :param Q: quaternion tensor of shape (b, d) where b is the batch-dimension, d is the feature dimension
    :return: real matrix representation of the quaternion

    for every quaternion matrix Q \in H^{b x d} with Q = Q_r + Q_i*i + Q_j*j + Q_k*k and Q_{r,i,j,k} \in R^{b x d}
    the quaternion matrix has a real-valued matrix representation
               _                     _
              | Q_r  -Q_i  -Q_j  -Q_k |
        R_Q = | Q_i   Q_r  -Q_k   Q_j |
              | Q_j   Q_k   Q_r  -Q_i |
              | Q_k  -Q_j   Q_i   Q_r |
               -                     -
    """
    if Q.dim() == 1:  # only one quaternion vector
        Q = Q.unsqueeze(dim=0)  # add batch-dimension

    row1 = torch.cat([Q.r, -Q.i, -Q.j, -Q.k], dim=-1)
    row2 = torch.cat([Q.i, Q.r, -Q.k, Q.j], dim=-1)
    row3 = torch.cat([Q.j, Q.k, Q.r, -Q.i], dim=-1)
    row4 = torch.cat([Q.k, -Q.j, Q.i, Q.r], dim=-1)

    RQ = torch.cat([row1, row2, row3, row4], dim=0)
    return RQ


def get_complex_matrix_representation(Q: QTensor) -> np.ndarray:
    r"""
    Returns the unique complex representation of a quaternion
    Let q = a*1 + b*i_1 + c*i_2 + d*i_3 be a quaternion number, with a,b,c,d \in R and (i_1,i_2,i_3) being the imaginary
    components (i,j,k) respectively.

    Then q can be decomposed into a complex 2X2 matrix.
    Let z = a + ib and w = c + id
    Then the complex representatin is

          | z        w      |
    C_Q = | -w.conj  z.conj |

    where z.conj is the conjugate of z, i.e. z.conj = a - ib

    :param Q: Quaternion tensor of shape [N x M]
    :return: Complex nd.ndarray of shape [2N X 2M]

    """
    z_r = Q.r
    z_i = Q.i
    # use numpy complex numbers
    z = z_r.detach().numpy() + 1j*z_i.detach().numpy()
    w_r = Q.j
    w_i = Q.k
    # use numpy complex numbers
    w = w_r.detach().numpy() + 1j*w_i.detach().numpy()

    row1 = np.hstack([z, w])
    row2 = np.hstack([-w.conjugate(), z.conjugate()])
    quaternion_cmplx = np.vstack([row1, row2])

    return quaternion_cmplx


def hamilton_product_Wq(W: QTensor, q: QTensor) -> QTensor:
    """
    Computes the Hamilton product W*q where q is a batch tensor of quaternions
    out = W @ q

    # W has shape (out_channels, in_channels)
    # q has shape (batch_size, in_channels)
    """
    out = W @ q.t()  # (out_channels, batch_size)
    out = out.t()  # (batch_size, out_channels)
    return out


def hamilton_product_qW(W: QTensor, q: QTensor) -> QTensor:
    """
    Computes the reverse Hamilton product q*W where q is a batch tensor of quaternions and W is a quaternion matrix
    out = q @ W
    # W has shape (out_channels, in_channels)
    # q has shape (batch_size, in_channels)
    """
    out = q @ W.t()  # (batch_size, out_channels)
    return out
