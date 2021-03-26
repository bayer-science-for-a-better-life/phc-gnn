import torch

""" 
The quaternion QR decomposition is based on this paper:
Real structure-preserving algorithms of Householder based transformations for quaternion matrices (Li et al. 2016)
https://www.sciencedirect.com/science/article/pii/S0377042716301583
"""


def RealP(A1: torch.Tensor, A2: torch.Tensor, A3: torch.Tensor, A4: torch.Tensor) -> torch.Tensor:
    r"""
    A method for generating a 4m x 4n real representation of matrix A = A1 + A2*i + A3*j + A4*k
    """
    # reshape as column vector if only one-dimension
    if A1.dim() == 1:
        c = len(A1)
        A1, A2, A3, A4 = A1.reshape(c, 1), A2.reshape(c, 1), A3.reshape(c, 1), A4.reshape(c, 1)
    row1 = torch.cat([A1, -A2, -A3, -A4], dim=1)
    row2 = torch.cat([A2, A1, -A4, A3], dim=1)
    row3 = torch.cat([A3, A4, A1, -A2], dim=1)
    row4 = torch.cat([A4, -A3, A2, A1], dim=1)
    real_repr = torch.cat([row1, row2, row3, row4], dim=0)

    assert real_repr.size(0) == 4 * A1.size(0) and real_repr.size(1) == 4 * A1.size(1)
    return real_repr


def quat_householder(x1: torch.Tensor, x2: torch.Tensor,
                     x3: torch.Tensor, x4: torch.Tensor, n: int) -> (torch.Tensor, float):
    r"""
    get quaternion householder matrix x has either shape (k,) or (k,1)
    """
    # reshape x1, x2, x3, x4 as column vectors in case their shape is (k, )
    x1, x2, x3, x4 = x1.view(-1, 1), x2.view(-1, 1), x3.view(-1, 1), x4.view(-1, 1)
    u1 = torch.cat([x1, x2, x3, x4], dim=1).to(x1.dtype)
    u1 = u1[:n]  # or u1[:(n-1)] ? The algorithm is adapted from MatLab...
    aa = torch.cat([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()], dim=0).to(x1.dtype).norm(p=2).item()
    xx = torch.cat([x1[0], x2[0], x3[0], x4[0]], dim=0).to(x1.dtype).norm(p=2).item()
    if xx == 0.0:
        alpha1 = aa * torch.Tensor([1, 0, 0, 0]).to(x1.dtype)
    else:
        alpha1 = -(aa / xx) * torch.cat([x1[0], x2[0], x3[0], x4[0]], dim=0).to(x1.dtype).T

    u1[0, :] = u1[0, :] - alpha1
    beta1 = 1 / (aa * (aa + xx))
    u = RealP(u1[:, 0].view(-1, 1), u1[:, 1].view(-1, 1),
              u1[:, 2].view(-1, 1), u1[:, 3].view(-1, 1))
    u = u.to(x1.dtype)

    return u, beta1


def GRSGivens(g1: torch.Tensor, g2: torch.Tensor,
              g3: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
    if g2.item() == g3.item() == g4.item() == 0.0:
        G1 = torch.eye(4, dtype=g1.dtype)
    else:
        G1 = RealP(g1, g2, g3, g4)
        gnorm = torch.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2 + g4 ** 2)
        G1 /= gnorm

    return G1


def quat_QR(A1: torch.Tensor, A2: torch.Tensor, A3: torch.Tensor, A4: torch.Tensor,
            givens: bool = False) -> (torch.Tensor, torch.Tensor):

    # cast to torch.float64 for better precision
    B1 = torch.cat([A1, A2, A3, A4], dim=0).to(A1.dtype)  # row concat
    m = A1.size(0)
    n = A1.size(1)
    Q = [torch.eye(m, dtype=A1.dtype)] * 4
    # Q = RealP(*Q)
    Q = torch.cat(Q, dim=0)
    for j in range(0, n):
        x1 = B1[0 * m: 1 * m, j]
        x2 = B1[1 * m: 2 * m, j]
        x3 = B1[2 * m: 3 * m, j]
        x4 = B1[3 * m: 4 * m, j]
        x1 = x1[j:m]
        x2 = x2[j:m]
        x3 = x3[j:m]
        x4 = x4[j:m]

        u, beta1 = quat_householder(x1,
                                    x2,
                                    x3,
                                    x4, n=m - j)


        indices = [i for i in range(j, m)] + [i for i in range(j + m, 2 * m)] + [i for i in range(j + 2 * m, 3 * m)] + [
            i for i in range(j + 3 * m, 4 * m)]

        # in place
        B1[indices, :] = B1[indices, :] - (beta1 * u) @ (u.T @ B1[indices, :])
        Q[indices, :] = Q[indices, :] - (beta1 * u) @ (u.T @ Q[indices, :])

        if givens:
            G = GRSGivens(B1[0 * m, j].reshape(-1, 1),
                          B1[1 * m, j].reshape(-1, 1),
                          B1[2 * m, j].reshape(-1, 1),
                          B1[3 * m, j].reshape(-1, 1))

            for t in range(0, j):
                indices = [t, t + m, t + 2 * m, t + 3 * m]
                B1[indices, :] = G.T @ B1[indices, :]


    return Q, B1