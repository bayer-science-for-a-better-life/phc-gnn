import torch


def kronecker_product_single(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    assert A.dim() == B.dim() == 2
    A_height, A_width = A.size()
    B_height, B_width = B.size()
    out_height = A_height * B_height
    out_width = A_width * B_width

    tiled_B = B.repeat(A_height, A_width)
    expanded_A = (
        A.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, B_height, B_width, 1)
            .view(out_height, out_width)
    )

    # apply hadamard product on expansions
    kron = expanded_A * tiled_B
    return kron


def _kronecker_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == B.dim() == 2
    return torch.einsum("ab,cd->acbd", A, B).reshape(
        A.size(0) * B.size(0), A.size(1) * B.size(1)
    )


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                    A.size(1)*B.size(1),
                                                    A.size(2)*B.size(2)
                                                    )
    return res



"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out