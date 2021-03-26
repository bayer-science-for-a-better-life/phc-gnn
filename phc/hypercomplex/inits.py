import torch
import numpy as np
from scipy.stats import chi


def unitary_init(phm_dim: int, in_features: int, out_features: int, low: int = 0, high: int = 1):
    v = [torch.FloatTensor(in_features, out_features).zero_()]
    for i in range(phm_dim-1):
        v.append(torch.FloatTensor(in_features, out_features).uniform_(low, high))
    v = torch.stack(v, dim=0)  # [phm_dim, in_feat, out_feat]
    vnorm = v.norm(p=2, dim=0)  # [in_feat, out_feat]
    v = v / vnorm
    return v


def phm_init(phm_dim: int, in_features: int, out_features:int,
             low: int = 0, high: int = 1, criterion: str = 'glorot', transpose: bool = True):

    fan_in = in_features
    fan_out = out_features

    if criterion == 'glorot':
        s = np.sqrt(2. / (phm_dim * (fan_in + fan_out)))
    elif criterion == 'he':
        s = np.sqrt(2. / (phm_dim * fan_in))
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    kernel_shape = (in_features, out_features)
    magnitude = torch.from_numpy(chi.rvs(phm_dim, loc=0, scale=s, size=kernel_shape)).to(torch.float32)
    # purely imaginary vectormap
    v = unitary_init(phm_dim=phm_dim, in_features=in_features, out_features=out_features,
                     low=low, high=high)

    theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=kernel_shape)).to(torch.float32)

    weight = [magnitude * torch.cos(theta)]
    for vs in v[1:]:
        weight.append(magnitude * vs * torch.sin(theta))

    weight = torch.stack(weight, dim=0)
    if transpose:
        weight = weight.permute(0, 2, 1)
    return weight
