import torch
import math
from typing import Union

from phc.quaternion.algebra import QTensor



class IntegerEncoder(torch.nn.Module):
    """
    Implements the IntegerEncoder that takes integer-based features as input where the final embedding
    is the sum over the single embeddings.
    For example x = [x1, x2, x3, x4] where xi \in [0,...,max_xi] \forall i in [1,2,3,4]
    x1 gets transformed into embed(x1) \in R^out_dim
    x2 gets transformed into embed(x2) \in R^out_dim
    ...
    the final embedding is the sum over the embed(xi)
    """
    def __init__(self, out_dim: int, input_dims: list, combine: str = "sum") -> None:
        """
        """
        super(IntegerEncoder, self).__init__()
        assert combine in ["sum", "concat"]
        self.combine = combine
        self.out_dim = out_dim
        self.input_dims = input_dims
        self.embeddings = torch.nn.ModuleList()
        for embed_dim in input_dims:
            self.embeddings.append(torch.nn.Embedding(embed_dim, out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data, gain=math.sqrt(2))

    def get_number_of_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def print_number_of_params(self):
        return f"Module {self.__class__.__name__} has {self.get_number_of_params()} trainable parameters."


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if self.combine == "sum":
            out = 0
            for i in range(x.size(1)):
                out += self.embeddings[i](x[:, i])
        else:
            out = []
            for i in range(x.size(1)):
                out.append(self.embeddings[i](x[:, i]))
            out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return "{}(out_dim={}, input_dims={}, combine='{}')".format(self.__class__.__name__,
                                                                    self.out_dim, self.input_dims, self.combine)


class QuaternionEncoder(torch.nn.Module):
    def __init__(self, out_dim: int, input_dims: Union[list, int], combine: str = "sum"):
        super(QuaternionEncoder, self).__init__()
        self.input_dims = input_dims
        self.out_dim = out_dim
        self.combine = combine
        if isinstance(input_dims, list):
            self.r = IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
            self.i = IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
            self.j = IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
            self.k = IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
        elif isinstance(input_dims, int):
            self.r = torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True)
            self.i = torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True)
            self.j = torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True)
            self.k = torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True)
        else:
            print(f"Must insert datatype int or list. Data type {type(input_dims)} was inserted as `input_dims`.")
            raise ValueError

        self.reset_parameters()

    def reset_parameters(self):
        self.r.reset_parameters(), self.i.reset_parameters(), self.j.reset_parameters(), self.k.reset_parameters()

    def forward(self, x: torch.Tensor) -> QTensor:
        return QTensor(self.r(x),  self.i(x), self.j(x), self.k(x))

    def __repr__(self):
        return '{}(in_dims={}, out_dim={}, combine="{}")'.format(self.__class__.__name__,
                                                                 self.input_dims,
                                                                 self.out_dim,
                                                                 self.combine)



class NaiveQuaternionEncoder(torch.nn.Module):
    def __init__(self, out_dim: int, input_dims: Union[list, int], combine: str = "sum"):
        super(NaiveQuaternionEncoder, self).__init__()
        self.input_dims = input_dims
        self.out_dim = out_dim
        self.combine = combine
        if isinstance(input_dims, list):
            self.encoder = IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
        elif isinstance(input_dims, int):
            self.encoder = torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True)
        else:
            print(f"Must insert datatype int or list. Data type {type(input_dims)} was inserted as `input_dims`.")
            raise ValueError

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, x: torch.Tensor) -> QTensor:
        encoded = self.encoder(x)
        return QTensor(encoded.clone(), encoded.clone(), encoded.clone(), encoded.clone())

    def __repr__(self):
        return '{}(in_dims={}, out_dim={}, combine="{}")'.format(self.__class__.__name__,
                                                                 self.input_dims,
                                                                 self.out_dim,
                                                                 self.combine)
