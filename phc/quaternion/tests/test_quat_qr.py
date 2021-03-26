import torch
import unittest
import pytest

from phc.quaternion.qr import quat_QR, RealP


class TestQR(unittest.TestCase):
    def test_qr_decomposition(self):

        p = 16

        Ar, Ai, Aj, Ak = torch.randn(p, p), torch.randn(p, p), torch.randn(p, p), torch.randn(p, p)

        Q, R = quat_QR(Ar, Ai, Aj, Ak)
        Q_scaled = Q / 2.0

        Qr, Qi, Qj, Qk = Q_scaled.split(p, dim=0)

        Q_real_representation = RealP(Qr, Qi, Qj, Qk)
        eye = torch.eye(Q_real_representation.size(0))
        unitary = Q_real_representation @ Q_real_representation.t()
        diff = torch.abs(eye - unitary).sum().item()
        assert diff < 0.001
        assert abs(torch.mean(torch.diag(unitary)).item()) < 1.0 + 1e-6
