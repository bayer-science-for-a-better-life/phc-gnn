# Parameterized Hypercomplex (PHC) Modules

This subdirectory consists of the python files to costruct a parameterized hypercomplex GNN using the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) open-source library.
Basicaly, this subdirectory can be seen as a generalization of the `quaternion/` subdirectory, that includes the implementations shaped on the quaternion algebra.  
Here, we generalize to algebra dimension with `n` components, as opposed to the quaternion-algebra where `n=4`.  
With the parameterized hypercomplex GNN, we want to construct our GNNs that utilize the parameterized hypercomplex (PHM) layer<a href="#note1" id="note1ref"><sup>1</sup></a>.
The PHM affine transformation is defined in `hypercomplex/layers.py` in the `PHMLinear` class.
Basically, this class can be seen as a replacement of `torch.nn.Linear` to include the parameterized hypercomplex multiplication layer that inherently includes a weight-sharing mechanism.

All files in this directory can be seen as base-modules, to construct PHC NNs, 
while the `hypercomplex/undirectional/` subdirectory consists of the implementation for the PHC-GNN.  
We have included a few unit tests in the `hypercomplex/tests` subdirectory. To run them, do the following:
```
cd hypercomplex
pytest tests/
```

### Overview
Here we list some details of the basic modules used for PHC NNs.
* `aggregator.py`: includes the implementation of the Softattention graph-pooling module to process node-embeddings of a graph to a single graph-embedding that can be used by a downstream predictor.
* `downstream.py`: includes the implementation of a PHC feed-forward neural network to process the graph-embedding.
* `encoder.py`: includes the implementation to encode raw node- and edge-features to "PHC"-features.
* `inits.py`: includes the weight-initialization mainly taken and adapted from: [Orkis-Research/Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py#L583)
* `kronecker.py`: includes our custom implementation of the Kronecker product. Note that this implementation might be sub-optimal. Since we use the stable PyTorch version `1.7.1`, we cannot use the CUDA implementation of `torch.kron` merged into the master branch. [(see PR #45358)](https://github.com/pytorch/pytorch/pull/45358)
* `layers.py`: includes the implementation of the `PHMLinear` class to apply PHC affine transformations. Additionally, the file includes the implementation of a simple 2-layer MLP.
* `norm.py`: includes the implementation of PHC batch-normalization (BN).
Here, we only implement the naive BN variant, that applies batch-normalization on each separate real and imaginary unit, i.e., separately on each component of the second axis of `[N, n, k]`.
* `pooling.py`: includes the graph-pooling module to process node-embeddings from the PHC-GNN.
* `regularization.py`: includes the regularization on PHC weight matrices as well as the contribution matrices. 
* `utils.py`: includes the implementation of the initialization for the contribution matrices in the `PHMLinear` layer.
* `undirectional/`: includes the implementation of the PHC-GNN by utilizing PyTorch Geometrics `MessagePassing` base-class to construct the PHC messagepassing-layer. The final  PHC-GNNs are implemented in `undirectional/models.py`.
* `tests/`: includes unit-tests for the base implementations in this subdirectory.


#### References:
<a id="note1" href="#note1ref"><sup>1</sup></a>: [A. Zhang, Y. Tay, S. Zhang, A. Chan, A. T. Luu, S. C. Hui, and J. Fu
Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters.
In Proceedings of the 9th International Conference on Learning Representations (ICLR, Spotlight), 2021](https://openreview.net/forum?id=rcQdycl0zyk)