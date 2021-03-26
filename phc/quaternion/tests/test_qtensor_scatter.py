import unittest
import pytest
import torch
from torch_scatter import scatter_sum
from torch_geometric.nn import global_add_pool

from phc.quaternion.algebra import QTensor


class TestQTensorScatter(unittest.TestCase):
    def test_qtensor_scatter_idx(self):

        row_ids = 1024
        idx = torch.randint(low=0, high=256, size=(row_ids,), dtype=torch.int64)
        p = 64
        x = QTensor(*torch.randn(4, row_ids, p))

        x_tensor = x.stack(dim=1)
        assert x_tensor.size() == torch.Size([row_ids, 4, p])

        x_aggr = scatter_sum(src=x_tensor, index=idx, dim=0, dim_size=x_tensor.size(0))

        assert x_aggr.size() == x_tensor.size()
        x_aggr = x_aggr.permute(1, 0, 2)
        q_aggr = QTensor(*x_aggr)

        r = scatter_sum(x.r, idx, dim=0, dim_size=x.size(0))
        i = scatter_sum(x.i, idx, dim=0, dim_size=x.size(0))
        j = scatter_sum(x.j, idx, dim=0, dim_size=x.size(0))
        k = scatter_sum(x.k, idx, dim=0, dim_size=x.size(0))
        q_aggr2 = QTensor(r, i, j, k)

        assert q_aggr == q_aggr2


    def test_scatter_batch_idx(self):

        n_graphs = 128
        n_nodes = 2048
        idx = torch.randint(low=0, high=n_graphs, size=(n_nodes,), dtype=torch.int64)
        p = 64
        x = QTensor(*torch.randn(4, n_nodes, p))

        x_tensor = x.stack(dim=1)
        assert x_tensor.size() == torch.Size([n_nodes, 4, p])

        x_aggr = scatter_sum(src=x_tensor, index=idx, dim=0)
        x_aggr2 = global_add_pool(x_tensor, batch=idx)
        assert torch.allclose(x_aggr, x_aggr2)


        x_aggr = x_aggr.permute(1, 0, 2)
        q_aggr = QTensor(*x_aggr)

        r = scatter_sum(x.r, idx, dim=0)
        i = scatter_sum(x.i, idx, dim=0)
        j = scatter_sum(x.j, idx, dim=0)
        k = scatter_sum(x.k, idx, dim=0)
        q_aggr2 = QTensor(r, i, j, k)

        assert q_aggr == q_aggr2
        assert torch.allclose(x_aggr[0], r)
        assert torch.allclose(x_aggr[1], i)
        assert torch.allclose(x_aggr[2], j)
        assert torch.allclose(x_aggr[3], k)


        r1 = global_add_pool(x.r, idx)
        i1 = global_add_pool(x.i, idx)
        j1 = global_add_pool(x.j, idx)
        k1 = global_add_pool(x.k, idx)
        q_aggr3 = QTensor(r1, i1, j1, k1)

        assert q_aggr == q_aggr2 == q_aggr3
