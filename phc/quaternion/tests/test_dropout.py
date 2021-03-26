import torch
import unittest
import pytest

from phc.quaternion.algebra import QTensor
from phc.quaternion.norm import QuaternionNorm
from phc.quaternion.layers import quaternion_dropout


from itertools import permutations


class TestDropout(unittest.TestCase):

    def test_quaternion_dropout(self):
        batch_size = 128
        in_features = 32
        same = False
        p = 0.3
        q = QTensor(*torch.randn(4, batch_size, in_features))
        q_dropped = quaternion_dropout(q=q, p=p, training=True, same=same)

        q_tensor = q.stack(dim=0)
        q_dropped_tensor = q_dropped.stack(dim=0)
        # check that "on"-indices are the same when retrieving the data
        ids = (q_dropped_tensor != 0.0)
        q_on = q_tensor[ids]
        q_dropped_on = q_dropped_tensor[ids]
        q_dropped_on *= (1-p)  # rescaling

        assert torch.allclose(q_on, q_dropped_on)

        same = True
        q = QTensor(*torch.randn(4, batch_size, in_features))
        q_dropped = quaternion_dropout(q=q, p=p, training=True, same=same)

        q_tensor = q.stack(dim=0)
        q_dropped_tensor = q_dropped.stack(dim=0)
        # rescaling
        q_dropped_tensor *= (1-p)

        # check if quaternion-component axis is really 0 among all components
        ids = [(x != 0.0).to(torch.float32) for x in q_dropped_tensor]

        for a, b in permutations(ids, 2):
            assert torch.allclose(a, b)