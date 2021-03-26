# Quaternion Modules
This subdirectory consists of the python files to costruct a quaternion GNN using the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) open-source library.
The implementation of the basis algebra operations on `torch.Tensors` are modified in `quaternion/algebra.py` into the `QTensor` class, that includes the addition and multiplication as defined in the quaternion algebra.
  
The quaternion affine transformation which is heavily utilized in quaternion-valued neural networks (NNs) is defined in `quaternion/layers.py` in the `QLinear` class.
Basically, this class can be seen as a replacement of `torch.nn.Linear` to include the Hamilton Product / Quaternion Affine Transformation that inherently includes a weight-sharing mechanism.

All files in this directory can be seen as base-modules, to construct quaternion-valued NNs, while the `quaternion/undirectional/` subdirectory consists of the implementation for the quaternion GNN.  
We have included a few unit tests in the `quaternion/tests` subdirectory. To run them, do the following:
```
cd quaternion
pytest tests/
```

### Overview
Here we list some details of the basic modules used for quaternion-valued NNs.
* `algebra.py`: consists of the base implementations of primitive operations in the quaternion algebra.
* `activations.py`: consists of activation functions that can process objects of the `QTensor` class.
We apply deplot the activation functions in a **split**-fashion. That means, given a quaternion-valued tensor of shape `[N, 4, k]` 
we apply the non-linearity separately on each dimension of the second axis. Technically, this can be seen as reshaping the tensor to the shape `[N, 4*k]` 
and applying the non-linearity as commonly known in feed-forward NNs.
* `aggregator.py`: includes the implementation of the Softattention graph-pooling module to process node-embeddings of a graph to a single graph-embedding that can be used by a downstream predictor.
* `downstream.py`: includes the implementation of a quaternion-valued feed-forward neural network to process the graph-embedding.
* `encoder.py`: includes the implementation to encode raw node- and edge-features to "quaternion-valued" features.
* `inits.py`: includes the weight-initialization mainly taken and adapted from<a href="#note1" id="note1ref"><sup>1</sup></a>: [Orkis-Research/Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py#L583)
* `layers.py`: includes the implementation of the `QLinear` class to apply affine transformations on quaternion-valued vectors/matrices. Additionally, the file includes the implementation of a simple 2-layer MLP.
* `norm.py`: includes the implementation of quaternion batch-normalization (BN). We provide two options for the quaternion BN: (1) based on the whitening approach that is computationally more expensive, as it requires the computation of cholesky decompositions. Option (2) is the naive variant, that applies batch-normalization on each separate real and imaginary unit, i.e., separately on each component of the second axis of `[N, 4, k]`.
* `pooling.py`: includes the graph-pooling module to process node-embeddings from the quaternion-GNN.
* `qr.py`: includes an implementation of the QR decomposition of quaternion-valued matrices. The incentive for having the QR-decomposition was to initialize the quaternion-valued weight-matrices with orthogonal matrices.
* `regularization.py`: includes the regularization on quaternion-valued weights. 
* `undirectional/`: includes the implementation of the quaternion GNN by utilizing PyTorch Geometrics `MessagePassing` base-class to construct the quaternion messagepassing-layer. The final quaternion GNNs are implemented in `undirectional/models.py`.
* `tests/`: includes unit-tests for the base implementations in this subdirectory.

#### References:
<a id="note1" href="#note1ref"><sup>1</sup></a>: [Parcollet, T., Ravanelli, M., Morchid, M., Linarès, G., Trabelsi, C., Mori, R., & Bengio, Y. (2019). Quaternion Recurrent Neural Networks.
In Proceedings of the 7th International Conference on Learning Representations (ICLR, Poster), 2019](https://arxiv.org/pdf/1806.04418.pdf)
