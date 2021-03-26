import unittest
import pytest
import torch

from phc.quaternion.algebra import QTensor


class TestQTensorMul(unittest.TestCase):
    def test_simple_mul(self):

        # real quaternion tensor addition
        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0, -1.0]),
                     i=torch.tensor([2.0, 2.0, 3.0, -2.0]),
                     j=torch.tensor([2.0, 1.0, 0.0, 1.5]),
                     k=torch.tensor([5.0, 4.0, 3.0, 5.0]))

        t2 = QTensor(r=torch.tensor([2.0, 3.0, 4.0, -0.5]),
                     i=torch.tensor([3.0, 1.0, 2.0, 2.0]),
                     j=torch.tensor([1.0, 0.0, 1.0, -3]),
                     k=torch.tensor([4.0, 3.0, 2.0, -4]))

        # t1 has shape (4) and t2 has shape (4)
        # hamilton product is done element-wise similar to the hadamard product which applies the
        # multiplication element-wise for the vector component
        t3 = t1 * t2  # has shape (4)

        # alternative calculation of quaternion representation via dot product and cross product
        # https://www.3dgep.com/understanding-quaternions/#Quaternion_Products
        t1_vec = []
        t2_vec = []
        # retrieve the vector/imaginary parts ijk
        for s1, s2 in zip(t1, t2):
            t1_vec.append([s1.i, s1.j, s1.k])
            t2_vec.append([s2.i, s2.j, s2.k])


        t1_vec = torch.tensor(t1_vec)  # (4,3)
        t2_vec = torch.tensor(t2_vec)  # (4,3)
        cross_product = torch.cross(t1_vec, t2_vec, dim=1)
        cross_product_i = t1.j * t2.k - t2.j * t1.k
        cross_product_j = t1.k * t2.i - t2.k * t1.i
        cross_product_k = t1.i * t2.j - t2.i * t1.j
        cross_product_manual = torch.cat([cross_product_i.view(-1, 1),
                                          cross_product_j.view(-1, 1),
                                          cross_product_k.view(-1, 1)], dim=1)
        assert torch.allclose(cross_product, cross_product_manual)

        # get the real part from
        r = t1.r * t2.r - torch.sum(t1_vec * t2_vec, dim=1)  # [4]
        # get the vector/imaginary parts
        ijk = t1.r.view(-1, 1) * t2_vec + t2.r.view(-1, 1) * t1_vec + cross_product  # [4,3]
        r = r.unsqueeze(dim=1)
        rijk = torch.cat([r, ijk], dim=1).t()  # (4,4)

        t3_tensor = t3.stack(dim=0)

        assert torch.allclose(t3_tensor, rijk)

        t3 = t1 * t2
        t3_not = t2 * t1
        assert not t3 == t3_not


    def test_simple_mul_left_right(self):


        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0, -1.0]),
                     i=torch.tensor([2.0, 2.0, 3.0, -2.0]),
                     j=torch.tensor([2.0, 1.0, 0.0, 1.5]),
                     k=torch.tensor([5.0, 4.0, 3.0, 5.0])).requires_grad_()

        t2 = torch.tensor([5.0, -1.0, 2.0, -2.0])

        t3_left = t1 * t2
        t3_right = t2 * t1

        assert t3_left == t3_right

        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0, -1.0]),
                     i=torch.tensor([2.0, 2.0, 3.0, -2.0]),
                     j=torch.tensor([2.0, 1.0, 0.0, 1.5]),
                     k=torch.tensor([5.0, 4.0, 3.0, 5.0])).requires_grad_()


        t1 *= 2.0
        t3 = QTensor(r=torch.tensor([2.0, 4.0, 6.0, -2.0]),
                     i=torch.tensor([4.0, 4.0, 6.0, -4.0]),
                     j=torch.tensor([4.0, 2.0, 0.0, 3.0]),
                     k=torch.tensor([10.0, 8.0, 6.0, 10.0]))
        res = t1 - t3
        res0 = QTensor.zeros(4)
        assert res == res0



        dropout_mask = (torch.empty(16, 9).uniform_() > 0.1).float()
        q = QTensor(*torch.randn(4, 16, 9))
        q_dropped0 = dropout_mask * q
        q_dropped1 = q * dropout_mask

        assert q_dropped0 == q_dropped1

        r, i, j, k = q.r * dropout_mask, q.i * dropout_mask, q.j * dropout_mask, q.k * dropout_mask
        q_dropped_manually = QTensor(r, i, j, k)

        assert q_dropped0 == q_dropped1 == q_dropped_manually


    def test_inverse_mul(self):


        t1 = QTensor(r=torch.tensor([1.0, 2.0, 3.0, -1.0]),
                     i=torch.tensor([2.0, 2.0, 3.0, -2.0]),
                     j=torch.tensor([2.0, 1.0, 0.0, 1.5]),
                     k=torch.tensor([5.0, 4.0, 3.0, 5.0]))

        unit_real = t1 * t1.inverse()
        assert torch.allclose(unit_real.r, torch.ones_like(unit_real.r))
        assert torch.allclose(unit_real.i, torch.zeros_like(unit_real.i))
        assert torch.allclose(unit_real.j, torch.zeros_like(unit_real.j))
        assert torch.allclose(unit_real.k, torch.zeros_like(unit_real.k))

        # (p*q)^-1 = q^-1 * p^-1
        t2 = QTensor(r=torch.tensor([2.0, 3.0, 4.0, -0.5]),
                     i=torch.tensor([3.0, 1.0, 2.0, 2.0]),
                     j=torch.tensor([1.0, 0.0, 1.0, -3]),
                     k=torch.tensor([4.0, 3.0, 2.0, -4]))

        out1 = (t1 * t2).inverse()
        out2 = t2.inverse() * t1.inverse()
        assert torch.allclose(out1.r, out2.r)
        assert torch.allclose(out1.i, out2.i)
        assert torch.allclose(out1.j, out2.j)
        assert torch.allclose(out1.k, out2.k)


    def test_mul_grad(self):
        x = QTensor(r=torch.tensor([1.0]),
                    i=torch.tensor([2.0]),
                    j=torch.tensor([2.0]),
                    k=torch.tensor([5.0])).requires_grad_()

        w = QTensor(r=torch.tensor([0.1]),
                    i=torch.tensor([0.2]),
                    j=torch.tensor([0.3]),
                    k=torch.tensor([0.4])).requires_grad_()

        # f(w,x) = w*x
        y = w * x

        # recall that multiplication is as follow
        # w * x = a + b*i + c*j + d*k, where
        #
        # (1) a = w_r*x_r - w_i*x_i - w_j*x_j - w_k*x_k
        # (2) b = w_i*x_r + w_r*x_i - w_k*x_j - w_j*x_k
        # (3) c = w_j*x_r + w_k*x_i + w_r*x_j - w_i*x_k
        # (4) d = w_k*x_r - w_j*x_i + w_i*x_j + w_r*x_k











