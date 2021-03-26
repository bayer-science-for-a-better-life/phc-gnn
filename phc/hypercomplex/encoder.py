import torch
from typing import Union

from phc.quaternion.encoder import IntegerEncoder


class PHMEncoder(torch.nn.Module):
    def __init__(self, out_dim: int, input_dims: Union[list, int], phm_dim: int, combine: str = "sum"):
        super(PHMEncoder, self).__init__()
        self.input_dims = input_dims
        self.out_dim = out_dim
        self.phm_dim = phm_dim
        self.combine = combine
        if isinstance(input_dims, list):
            self.encoders = torch.nn.ModuleList([IntegerEncoder(input_dims=input_dims, out_dim=out_dim, combine=combine)
                                                 for _ in range(phm_dim)])
        elif isinstance(input_dims, int):
            self.encoders = torch.nn.ModuleList(
                torch.nn.Linear(in_features=input_dims, out_features=out_dim, bias=True) for _ in range(phm_dim)
            )
        else:
            print(f"Must insert datatype int or list. Data type {type(input_dims)} was inserted as `input_dims`.")
            raise ValueError

        self.reset_parameters()

    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [enc(x) for enc in self.encoders]
        encoded = torch.stack(encoded, dim=1)  # [*, self.phm_dim, *]
        return encoded

    def __repr__(self):
        return '{}(in_dims={}, out_dim={},  phm_dim={}, combine="{}")'.format(self.__class__.__name__,
                                                                              self.input_dims,
                                                                              self.out_dim,
                                                                              self.phm_dim,
                                                                              self.combine)



class NaivePHMEncoder(torch.nn.Module):
    def __init__(self, out_dim: int, input_dims: Union[list, int], phm_dim: int, combine: str = "sum"):
        super(NaivePHMEncoder, self).__init__()
        self.input_dims = input_dims
        self.out_dim = out_dim
        self.phm_dim = phm_dim
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        encoded = [encoded.clone() for _ in range(self.phm_dim)]
        encoded = torch.stack(encoded, dim=1)
        return encoded


    def __repr__(self):
        return '{}(in_dims={}, out_dim={}, phm_dim={}, combine="{}")'.format(self.__class__.__name__,
                                                                             self.input_dims,
                                                                             self.out_dim,
                                                                             self.phm_dim,
                                                                             self.combine)


"""
dims = [10, 5, 9, 12, 10]
out_dim = 32
phm_dim = 3
encoder = PHMEncoder(out_dim=out_dim, input_dims=dims, phm_dim=phm_dim)
x = torch.randint(low=0, high=5, size=(512, len(dims)), dtype=torch.long)
y = encoder(x)

encoder2 = PHMEncoder(out_dim=16, input_dims=[6, 10, 5], phm_dim=phm_dim)

xx = torch.randint(low=0, high=4, size=(1024, 3), dtype=torch.long)
yy = encoder2(xx)
"""
