import torch
import numpy as np
import warnings

QUATERNION_MULTIPLICATION_RULE = {
    "A1": torch.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=torch.float32),
    "A2": torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]], dtype=torch.float32),
    "A3": torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]], dtype=torch.float32),
    "A4": torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]], dtype=torch.float32)
}


def get_quaternion_mul_rule(device: str = "cpu") -> list:
    matrices = [A.to(device) for A in QUATERNION_MULTIPLICATION_RULE.values()]
    return matrices


def get_complex_mul_rule() -> list:
    return [torch.tensor([[1, 0], [0, 1]], dtype=torch.float32),
            torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)]


def get_right_shift_permutation_matrix(phm_dim: int) -> torch.Tensor:
    permutation_matrix = np.eye(phm_dim, dtype=np.float32)
    permutation_matrix = np.roll(permutation_matrix, shift=1, axis=1)
    return torch.from_numpy(permutation_matrix)


def get_left_shift_permutation_matrix(phm_dim: int) -> torch.Tensor:
    permutation_matrix = np.eye(phm_dim, dtype=np.float32)
    permutation_matrix = np.roll(permutation_matrix, shift=-1, axis=1)
    return torch.from_numpy(permutation_matrix)


def right_cyclic_permutation(arr: list, power: int = 1) -> list:
    ids = len(arr)
    for _ in range(power):
        arr = [arr[-1]] + arr[:ids - 1]
    return arr


def left_cyclic_permutation(arr: list, power: int = 1) -> list:
    ids = len(arr)
    for _ in range(power):
        arr = arr[1:ids] + [arr[0]]
    return arr


def get_multiplication_matrices(phm_dim: int, type="standard"):
    assert type in ["standard", "random"]
    if type == "standard":
        assert phm_dim >= 1
        if phm_dim == 2:
            A_matrices = get_complex_mul_rule()
        elif phm_dim == 4:
            A_matrices = get_quaternion_mul_rule()
        else:
            permutation_matrix = get_right_shift_permutation_matrix(phm_dim)
            A_matrices = [torch.eye(phm_dim, dtype=torch.float32)]
            for i in range(1, phm_dim):
                values = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(phm_dim)], dtype=torch.float32)
                A = torch.diag(values)
                for j in range(i):
                    # right multiplication of A with P, where P is a permutation matrix, shifts the column of A.
                    A = A @ permutation_matrix
                A_matrices.append(A)
    elif type == "random":
        A_matrices = torch.FloatTensor(phm_dim, phm_dim, phm_dim).uniform_(-1, 1)
        A_matrices = [A for A in A_matrices]
    else:
        raise ValueError

    return A_matrices


"""
get_multiplication_matrices(1)
get_multiplication_matrices(2)
get_multiplication_matrices(3)
get_multiplication_matrices(4)
"""


def ensure_first_ax(x: torch.Tensor, type_dim: int = 4, type: str = "phm") -> torch.Tensor:
    """
    ensures that the input tensor `x` has `type_dim` as first axis.
    tensor must be of dimension 3, i.e. (a, b, c)
    some cases: (phm_dim, batch_size, feats) or (batch_size, phm_dim, feats)
    """
    assert type in ["phm", "batch"]
    assert x.dim() == 3
    shape = list(x.size())
    # get index where phm_dim is set / in case the phm_dim occurs more often, selects the first time of occurence but
    # will also raise a warning
    ids = np.where(np.array(shape) == type_dim)[0]
    if len(ids) > 1:
        warnings.warn(f"{type} occurs {len(ids)} times, of input tensor x with shape {x.size()}. \n"
                      f"Will used first time of occurence for permuting the axis.")
    ids = ids[0]
    if shape[0] == type_dim:
        return x
    else:
        full_order = list(range(x.dim()))
        full_order.pop(ids)
        reshaped = (ids, ) + tuple(full_order)
        x = x.permute(reshaped)
    return x


def phm_cat(tensors: list, phm_dim: int,  dim=-1) -> torch.Tensor:
    r""" Concatenates the list of hypercomplex tensors"""
    ntensors = len(tensors)
    feat_dim_sum = sum(tensor.size(1) for tensor in tensors)
    tensors = [t.reshape(t.size(0), phm_dim, t.size(1) // phm_dim).permute(1, 0, 2) for t in tensors]
    x = []
    for component in range(phm_dim):
        x.append(
            torch.cat([tensors[i][component] for i in range(ntensors)], dim=dim)
        )
    x = torch.cat(x, dim=dim)
    assert x.size(1) == feat_dim_sum

    return x


