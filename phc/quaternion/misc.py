import torch
from typing import Union

from phc.quaternion.algebra import QTensor


def row_diff(x: Union[torch.Tensor, QTensor]) -> float:
    r"""
    Implementation of the row_diff value from the paper "PAIRNORM: TACKLING OVERSMOOTHING IN GNNS" (ICLR 2020) by
    Lingxiao Zhao and Leman Akoglu

    The row-diff measure is the average of all pairwise distances between the node features (i.e., rows of
    the representation matrix) and quantifies node-wise oversmoothing.
    """
    if x.__class__.__name__ == "QTensor":
        x = x.stack(dim=0).permute(1, 0, 2)

    x = x.reshape(x.size(0), -1)
    pdist = torch.pdist(x, p=2)
    row_diff_value = (pdist.sum() / pdist.numel()).item()
    return row_diff_value


def col_diff(x: Union[torch.Tensor, QTensor]) -> float:
    r"""
    Implementation of the col_diff value from the paper "PAIRNORM: TACKLING OVERSMOOTHING IN GNNS" (ICLR 2020) by
    Lingxiao Zhao and Leman Akoglu

    The col-diff is the average of pairwise distances between (L1-normalized1) columns of the representation matrix
    and quantifies feature-wise oversmoothing.
    """
    if x.__class__.__name__ == "QTensor":
        x = x.stack(dim=0).permute(1, 0, 2)

    x = x.reshape(x.size(0), -1)
    col_norms = x.norm(p=1, dim=0, keepdim=True)
    xnormed = x / col_norms
    xnormed = xnormed.t()
    pdist = torch.pdist(xnormed, p=2)
    col_diff_value = (pdist.sum() / pdist.numel()).item()

    return col_diff_value

