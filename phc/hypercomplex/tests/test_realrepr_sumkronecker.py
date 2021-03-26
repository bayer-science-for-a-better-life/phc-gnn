import torch
import unittest
import pytest
import numpy as np


from phc.hypercomplex.utils import QUATERNION_MULTIPLICATION_RULE
from phc.hypercomplex.kronecker import kronecker_product

from phc.quaternion.qr import RealP


class TestQuatEqualKronecker(unittest.TestCase):
    def test_quaternion_rule_equal_kronecker_sum(self):

        out_features = 16
        in_features = 8
        Qr, Qi, Qj, Qk = torch.randn(out_features, in_features),\
                         torch.randn(out_features, in_features),\
                         torch.randn(out_features, in_features),\
                         torch.randn(out_features, in_features)


        # Get the real representation of the quaternion matrix by concatenating the corresponding components according
        # to the multiplication rule aka. real representation of quaternions
        QuaternionRealRepr = RealP(Qr, Qi, Qj, Qk)

        # over sum of kronecker product
        A1, A2, A3, A4 = QUATERNION_MULTIPLICATION_RULE["A1"], QUATERNION_MULTIPLICATION_RULE["A2"],\
                         QUATERNION_MULTIPLICATION_RULE["A3"], QUATERNION_MULTIPLICATION_RULE["A4"]
        QuaternionSumKronecker = kronecker_product(A1, Qr) + kronecker_product(A2, Qi) +\
                                 kronecker_product(A3, Qj) + kronecker_product(A4, Qk)

        assert torch.allclose(QuaternionRealRepr, QuaternionSumKronecker)