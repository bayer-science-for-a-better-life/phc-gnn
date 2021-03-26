import torch
import unittest
import pytest

from phc.quaternion.algebra import QTensor
from phc.quaternion.norm import QuaternionNorm
from phc.quaternion.layers import quaternion_dropout

from phc.hypercomplex.norm import PHMNorm
from phc.hypercomplex.layers import phm_dropout

from itertools import permutations



class TestBatchNorm(unittest.TestCase):
    def test_naive_batchnorm(self):


        # Naive batch-normalization for quaternions and phm (with phm_dim=4) should be the same.

        phm_dim = 4
        num_features = 16
        batch_size = 128
        quat_bn = QuaternionNorm(type="naive-batch-norm", num_features=num_features).train()
        phm_bn = PHMNorm(type="naive-batch-norm", phm_dim=phm_dim, num_features=num_features).train()

        distances = []
        for i in range(500):
            x = torch.randn(phm_dim, batch_size, num_features)
            x_q = QTensor(*x)
            x_p = x.permute(1, 0, 2).reshape(batch_size, -1)
            y_quat = quat_bn(x_q)
            y_phm = phm_bn(x_p)
            y_quat = y_quat.stack(dim=1)
            y_quat = y_quat.reshape(batch_size, -1)
            d = (y_phm - y_quat).norm().item()
            distances.append(d)

        assert sum(distances) == 0.0

        # eval mode uses running-mean and estimated weights + bias for rescaling and shifting.
        quat_bn = quat_bn.eval()
        phm_bn = phm_bn.eval()

        distances = []
        for i in range(500):
            x = torch.randn(4, batch_size, num_features)
            x_q = QTensor(*x)
            x_p = x.permute(1, 0, 2).reshape(batch_size, -1)
            y_quat = quat_bn(x_q)
            y_phm = phm_bn(x_p)
            y_quat = y_quat.stack(dim=1)
            y_quat = y_quat.reshape(batch_size, -1)
            d = (y_phm - y_quat).norm().item()
            distances.append(d)

        assert sum(distances)/len(distances) < 1e-4



class TestDropout(unittest.TestCase):

    def test_hypercomplex_dropout(self):
        batch_size = 128
        in_features = 32
        phm_dim = 4
        same = False
        p = 0.3
        x = torch.randn(batch_size, phm_dim, in_features)

        xx = x.reshape(batch_size, -1)
        xx_dropped = phm_dropout(x=xx, p=p, training=True, same=same, phm_dim=phm_dim)
        xx_dropped = xx_dropped.reshape(batch_size, phm_dim, in_features)

        # check that "on"-indices are the same when retrieving the data
        ids = (xx_dropped != 0.0)
        x_on = x[ids]
        x_dropped_on = xx_dropped[ids]
        x_dropped_on *= (1-p)  # rescaling

        assert torch.allclose(x_on, x_dropped_on)

        same = True
        x = torch.randn(batch_size, phm_dim, in_features)

        xx = x.reshape(batch_size, -1)
        xx_dropped = phm_dropout(x=xx, p=p, training=True, same=same, phm_dim=phm_dim)
        xx_dropped = xx_dropped.reshape(batch_size, phm_dim, in_features)

        # check that "on"-indices are the same when retrieving the data
        ids = (xx_dropped != 0.0)
        x_on = x[ids]
        x_dropped_on = xx_dropped[ids]
        x_dropped_on *= (1 - p)  # rescaling

        # check if phm_dim axis is really 0 among all components
        xx_dropped = xx_dropped.permute(1, 0, 2)
        ids = [(x != 0.0).to(torch.float32) for x in xx_dropped]


        for a, b in permutations(ids, 2):
            assert torch.allclose(a, b)


    def test_quaternion_hypercomplex_dropout(self):

        batch_size = 128
        in_features = 32
        phm_dim = 4
        same = False
        p = 0.3

        x = torch.randn(phm_dim, batch_size, in_features)
        q = QTensor(*x)


        q_dropped = quaternion_dropout(q, p=p, training=True, same=same)
        q_dropped = q_dropped.stack(dim=1)

        xx = x.permute(1, 0, 2).reshape(batch_size, -1)
        xx_dropped = phm_dropout(x=xx, p=p, training=True, same=same, phm_dim=phm_dim)
        xx_dropped = xx_dropped.reshape(batch_size, phm_dim, in_features)

        # check that the values, where quaternion-dropped and hypercomplex-dropped are "on", i.e. populated
        ids = (q_dropped != 0.0) * (xx_dropped != 0.0)
        on_q = q_dropped[ids]
        on_phm = xx_dropped[ids]
        assert torch.allclose(on_q, on_phm)