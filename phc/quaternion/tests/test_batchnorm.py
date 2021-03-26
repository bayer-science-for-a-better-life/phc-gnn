import torch
import unittest
import pytest
import numpy as np

from phc.quaternion.norm import QuaternionNorm
from phc.quaternion.algebra import QTensor


class TestBatchNorm(unittest.TestCase):
    def test_naive_batch_norm(self):

        batch_size = 128
        in_channels = 16
        q = QTensor(*torch.rand(4, batch_size, in_channels)*2.0 + 5)  # (128, 16)

        batch_norm = QuaternionNorm(num_features=in_channels, type="naive-batch-norm")
        batch_norm = batch_norm.train()

        y = batch_norm(q)

        # assert that r,i,j,k components all have mean 0 separately
        mean_r = torch.mean(y.r, dim=0)  # (16,)
        mean_i = torch.mean(y.i, dim=0)
        mean_j = torch.mean(y.j, dim=0)
        mean_k = torch.mean(y.k, dim=0)
        assert abs(mean_r.norm().item()) < 1e-5
        assert abs(mean_i.norm().item()) < 1e-5
        assert abs(mean_j.norm().item()) < 1e-5
        assert abs(mean_k.norm().item()) < 1e-5

        # assert that r,i,j,k components all have (biased) standard deviation of 1 separately
        std_r = torch.std(y.r, dim=0, unbiased=False)   # (16, )
        std_i = torch.std(y.i, dim=0, unbiased=False)
        std_j = torch.std(y.j, dim=0, unbiased=False)
        std_k = torch.std(y.k, dim=0, unbiased=False)
        assert abs(std_r - 1.0).sum().item() < 0.001
        assert abs(std_i - 1.0).sum().item() < 0.001
        assert abs(std_j - 1.0).sum().item() < 0.001
        assert abs(std_k - 1.0).sum().item() < 0.001

        # what about the covariance between for each quaternion number?
        y_stacked = y.stack(dim=0)  # (4, 128, 16)
        perm = y_stacked.permute(2, 0, 1)  # [16, 4, 1]
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]  # [16, 4, 4]

        eye_covs = torch.stack([torch.eye(4) for _ in range(cov.size(0))], dim=0)
        diffs = cov - eye_covs
        a = torch.abs(diffs).sum().item()

        assert np.round(a, 4) > 1.0



    def test_quaternion_batchnorm(self):
        batch_size = 128
        in_channels = 16
        q = QTensor(*torch.rand(4, batch_size, in_channels) * 2.0 + 5)  # (128, 16)

        batch_norm = QuaternionNorm(num_features=in_channels, type="q-batch-norm", **{"affine": False})
        batch_norm = batch_norm.train()

        y = batch_norm(q)  # (128, 16)

        #  each quaternion number has mean 0 and standard deviation 1
        y_stacked = y.stack(dim=0)  # (4, 128, 16)
        perm = y_stacked.permute(2, 0, 1)  # [16, 4, 1]

        # covariances of shape (16, 4, 4)
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]  # [16, 4, 4]

        eye_covs = torch.stack([torch.eye(4) for _ in range(cov.size(0))], dim=0)
        diffs = cov - eye_covs
        a = torch.abs(diffs).sum().item()
        assert 0.0001 < np.round(a, 4) < 0.05

        # check mean
        assert abs(y_stacked.mean(1).norm().item()) < 1e-4


