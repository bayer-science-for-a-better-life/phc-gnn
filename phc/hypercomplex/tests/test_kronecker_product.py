import torch
import unittest
import pytest

from phc.hypercomplex.kronecker import kronecker_product,\
    kronecker_product_single, kronecker_product_einsum_batched


class TestBatchedKronecker(unittest.TestCase):
    def test_batched_kronecker(self):

        phm_dim = 4
        out_feats = 16
        in_feats = 8
        A_l, B_l = [torch.randn(phm_dim, phm_dim) for _ in range(phm_dim)],\
                   [torch.randn(out_feats, in_feats) for _ in range(phm_dim)]

        A, B = torch.stack(A_l, dim=0), torch.stack(B_l, dim=0)

        single_kron = [kronecker_product_single(a, b) for a, b in zip(A_l, B_l)]
        single_kron = torch.stack(single_kron, dim=0)

        kron_einsum = kronecker_product_einsum_batched(A, B)

        kron = kronecker_product(A, B)

        assert torch.allclose(single_kron, kron)
        assert torch.allclose(single_kron, kron_einsum)
        assert torch.allclose(kron, kron_einsum)