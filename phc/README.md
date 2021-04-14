### PHC library that consists of the quaternion and hypercomplex modules

##### Examples on how to use the `PHMLinear` layer:

The following code explains how to initialize a PHM-layer that could replace the commonly used `torch.nn.Linear` module.
```python
import torch
from phc.hypercomplex.layers import PHMLinear

in_channels = 128
out_channels = 256

phm_dim = 4

device = "cuda:0" if torch.cuda.is_available() else "cpu"

phm_linear = PHMLinear(in_features=in_channels // phm_dim,
                       out_features=out_channels // phm_dim,
                       bias=True,
                       phm_dim=phm_dim).to(device)

standard_linear = torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=True).to(device)

def get_num_params(module: torch.nn.Module):
    return sum(m.numel() for m in module.parameters() if m.requires_grad)

print(f"PHM Linear has {get_num_params(phm_linear)} trainable parameters.")
# PHM Linear has 8512 trainable parameters.
print(f"Standard Linear has {get_num_params(standard_linear)} trainable parameters.")
# Standard Linear has 33024 trainable parameters.

x = torch.randn(512, in_channels).to(device)

y0 = phm_linear(x)
print(y0.shape) 
# torch.Size([512, 256])

y1 = standard_linear(x)
print(y1.shape) 
# torch.Size([512, 256])
```

As of now, the in- and output channels have to be divided by the `phm_dim` **before** instantiating the module. We will work on optimizing the code and include further documentation on usage.
In case you do not want to include so much of weight-sharing, decrease the `phm_dim`, e.g.:

```python
in_channels = 128
out_channels = 256
phm_dim= 2

phm_linear = PHMLinear(in_features=in_channels // phm_dim,
                       out_features=out_channels // phm_dim,
                       bias=True,
                       phm_dim=phm_dim).to(device)
print(f"PHM Linear has {get_num_params(phm_linear)} trainable parameters.")
#PHM Linear has 16648 trainable parameters.
```
