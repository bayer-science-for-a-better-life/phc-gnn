conda activate phc-gnn
echo "Installing Pytorch Geometric version 1.6.1  with Pytorch version 1.7.1 and CUDA 10.1"
# pyG wheels for torch 1.7.1 are the same as for torch 1.7.0 

pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_scatter-2.0.5-cp38-cp38-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_sparse-0.6.8-cp38-cp38-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_spline_conv-1.2.0-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==1.6.1
