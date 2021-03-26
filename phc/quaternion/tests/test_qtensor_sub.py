import unittest
import pytest
import torch

from phc.quaternion.algebra import QTensor


class TestQTensorSub(unittest.TestCase):
    def test_simple_sub(self):

        # real quaternion tensor subtraction
        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0]),
                     i=torch.tensor([2.0, 2.0, 3.0]),
                     j=torch.tensor([2.0, 1.0, 0.0]),
                     k=torch.tensor([5.0, 4.0, 3.0]))

        t2 = QTensor(r=torch.tensor([2.0, 3.0, 4.0]),
                     i=torch.tensor([3.0, 1.0, 2.0]),
                     j=torch.tensor([1.0, 0.0, 1.0]),
                     k=torch.tensor([4.0, 3.0, 2.0]))

        t3 = t1 - t2
        r, i, j, k = t3.r, t3.i, t3.j, t3.k

        assert torch.allclose(input=r, other=torch.tensor([-1.0, -1.0, -1.0]))
        assert torch.allclose(input=i, other=torch.tensor([-1.0, 1.0, 1.0]))
        assert torch.allclose(input=j, other=torch.tensor([1.0, 1.0, -1.0]))
        assert torch.allclose(input=k, other=torch.tensor([1.0, 1.0, 1.0]))

        #  quaternion tensor subtraction res = q - other, where other is only a float
        t2 = 2.0
        t3 = t1 - t2   # only subtracts to the real component
        r, i, j, k = t3.r, t3.i, t3.j, t3.k
        assert torch.allclose(input=r, other=torch.tensor([-1.0, 0.0, 1.0]))
        assert torch.allclose(input=i, other=t1.i)
        assert torch.allclose(input=j, other=t1.j)
        assert torch.allclose(input=k, other=t1.k)

        # quaternion tensor subtraction res = q + other, where other is a torch.Tensor
        t2 = torch.tensor([1.0, 2.0, 3.0])
        t3 = t1 - t2  # only subtracts to the real component
        r, i, j, k = t3.r, t3.i, t3.j, t3.k
        assert torch.allclose(input=r, other=torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(input=i, other=t1.i)
        assert torch.allclose(input=j, other=t1.j)
        assert torch.allclose(input=k, other=t1.k)

        # inplace subtraction
        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0]),
                     i=torch.tensor([2.0, 2.0, 3.0]),
                     j=torch.tensor([2.0, 1.0, 0.0]),
                     k=torch.tensor([5.0, 4.0, 3.0]))

        t2 = QTensor(r=torch.tensor([2.0, 3.0, 4.0]),
                     i=torch.tensor([3.0, 1.0, 2.0]),
                     j=torch.tensor([1.0, 0.0, 1.0]),
                     k=torch.tensor([4.0, 3.0, 2.0]))

        t3 = t1 - t2
        t1 -= t2

        assert torch.allclose(t1.r, t3.r)
        assert torch.allclose(t1.i, t3.i)
        assert torch.allclose(t1.j, t3.j)
        assert torch.allclose(t1.k, t3.k)

        # right subtraction full quaternionic tensors
        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0]),
                     i=torch.tensor([2.0, 2.0, 3.0]),
                     j=torch.tensor([2.0, 1.0, 0.0]),
                     k=torch.tensor([5.0, 4.0, 3.0]))

        t2 = QTensor(r=torch.tensor([2.0, 3.0, 4.0]),
                     i=torch.tensor([3.0, 1.0, 2.0]),
                     j=torch.tensor([1.0, 0.0, 1.0]),
                     k=torch.tensor([4.0, 3.0, 2.0]))
        t3 = t1 - t2
        t3_r = t2 - t1

        assert torch.allclose(t3.r, -t3_r.r)
        assert torch.allclose(t3.i, -t3_r.i)
        assert torch.allclose(t3.j, -t3_r.j)
        assert torch.allclose(t3.k, -t3_r.k)

        # right subtraction if other is a float
        t2 = 2.0
        t3 = t1 - t2
        t3_r = t2 - t1

        assert torch.allclose(t3.r, -t3_r.r)
        assert torch.allclose(t3.i, -t3_r.i)
        assert torch.allclose(t3.j, -t3_r.j)
        assert torch.allclose(t3.k, -t3_r.k)

        t2 = torch.tensor([1.0, 2.0, 3.0])
        t3 = t1 - t2
        t3_r = t2 - t1

        assert torch.allclose(t3.r, -t3_r.r)
        assert torch.allclose(t3.i, -t3_r.i)
        assert torch.allclose(t3.j, -t3_r.j)
        assert torch.allclose(t3.k, -t3_r.k)