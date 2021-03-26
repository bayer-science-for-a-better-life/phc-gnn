# Benchmark
We provide the scripts for the execution of the 6 benchmarks we used.
In total, we benchmarked our models on 3 datasets from the [Opengraph Benchmark (OGB)](https://github.com/snap-stanford/ogb)<a id="note1" href="#note1ref"><sup>1</sup></a>  and 3 datasets from [Benchmarking-GNNs](https://github.com/graphdeeplearning/benchmarking-gnns)<a id="note2" href="#note1ref"><sup>2</sup></a> , while our main experiments were conducted on the molecular graph property datasets
from OGB (i.e., ogbg-molhiv and ogbg-molpcba) and Benchmarking-GNNs (i.e., ZINC), which we also report in our main paper.

### Datasets
All datasets will be downloaded by the `ogb` and `torch_geometric` python packages when executing the `hiv, pcba, ppa` and `zinc, mnist and cifar10` scripts. Since we use PyTorch-Geometric to construct our PHC-GNN, the downloaded datasets will be further processed through the Dataloaders provided by Pytorch-Geometric.  
If not existent, the following directories in `benchmarks/dataset` will be created:

```
phc-gnn (root)
├── README.md
├── .gitignore
├── phc
|    ├── hypercomplex
|    |   └── ...
|    └── quaternion
|        └── ...
└── benchmarks
    ├── README.md
    ├── dataset
    │   └── ogbg_molhiv
    │   └── ogbg_molpcba
    .
    .
    │   └── CIFAR10
    ├── hiv
    │   ├── experiment1
    │   └── ...
    ├── pcba
    │   ├── experiment1
    │   └── ...
    .
    .
    ├── cifar10
    │   ├── experiment1
    │   └── ...
```
With the results for each experiment saved in `benchmarks/` corresponding sub-directory.

The 6 example bash-scripts include the shell calls to run a default PHC-4 GNN model on the 6 datasets.

To train the top-perfoming models as highlighted in our main article on `ogbg-molhiv, ogbg-molpcba, ZINC (100k)`
, execute the following (assuming GPU-ID 0 is available):

#### ogbg-molhiv
```
python train_hiv.py --input_embed_dim 200 --mp_units 200,200 --mlp_mp True\
   --dropout_mpnn 0.2,0.2 --mp_norm naive-batch-norm --d_units 128,32\
   --aggr_msg softmax --aggr_node softmax --dropout_dn 0.3,0.1 --phm_dim 4 --learn_phm True\
   --d_bn naive-batch-norm --type undirectional-phm-sc-add\
   --save_dir experiment1 --n_runs 10 --epochs 50\
   --w_init phm --c_init standard\
   --nworkers 0 --device 0 --lr 0.001 --patience 5 --factor 0.75 --weightdecay 0.1\
   --grad_clipping 2.0 --sc_type first
```

#### ogbg-molpcba
```
 python train_pcba.py --input_embed_dim 512 --mp_units 512,512,512,512,512,512,512 --mlp_mp False\
   --dropout_mpnn 0.1,0.1,0.1,0.1,0.1,0.1,0.1 --mp_norm naive-batch-norm --d_units 768,256\
   --aggr_msg sum --aggr_node sum --dropout_dn 0.3,0.2 --phm_dim 2 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir experiment1 --n_runs 5 --epochs 150\
   --w_init phm --c_init standard\
   --nworkers 0 --device 0 --lr 0.0005 --patience 5 --factor 0.75\
   --weightdecay 0.0001 --grad_clipping 2.0 --sc_type first
```

#### ZINC (100K model parameters)
```
 python train_zinc.py --input_embed_dim 180 --mp_units 200,200,200,200 --mlp_mp True\
   --dropout_mpnn 0,0,0,0 --mp_norm naive-batch-norm --d_units 200,128\
   --aggr_msg sum --aggr_node sum --dropout_dn 0.2,0.1 --phm_dim 5 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir phc8_random --n_runs 4 --epochs 1000\
   --w_init phm --c_init standard\
   --full_encoder True --nworkers 0 --device 0 --lr 0.001 --patience 10 --factor 0.5\
   --weightdecay 0.01 --grad_clipping 2.0 --sc_type last
```

If you want to increase the weight-sharing mechanism, simply increase the `--phm_dim` flag. For example, training a PHC-8 GNN model on the `molpcba` dataset with
random initialization for the contribution matrices (`--c_init random`):

#### ogbg-molpcba with PHC-8 model and random contribution matrices
```
 python train_pcba.py --input_embed_dim 512 --mp_units 512,512,512,512,512,512,512 --mlp_mp False\
   --dropout_mpnn 0.1,0.1,0.1,0.1,0.1,0.1,0.1 --mp_norm naive-batch-norm --d_units 768,256\
   --aggr_msg sum --aggr_node sum --dropout_dn 0.3,0.2 --phm_dim 8 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir tuned_setting --n_runs 5 --epochs 150\
   --w_init phm --c_init random\
   --nworkers 0 --device 0 --lr 0.0005 --patience 5 --factor 0.75\
   --weightdecay 0.0001 --grad_clipping 2.0 --sc_type first
```
This PHC-8 GNN will only consists of 689K trainable parameters, as opposed to the PHC-2 GNN with 1.69M trainable parameters.


### Tracking your training
To observe the training and validation performance of your models, we provide next to the `experimentX/run.log` file the Tensorboard loggings.
If you want to track the tensorboard logging (on localhost 8888) for a specific dataset, e.g. `hiv`, execute following command in a new shell
```
tensorboard --logdir benchmarks/hiv/ --port 8888
```

### Minimal running inference code
We provide three minimal examples in the `benchmarks/inference.ipynb` notebook to evaluate the models from the
`ogbg-molhiv, ogbg-molpcba, ZINC` datasets on the first training run.

### References
<a id="note1" href="#note1ref"><sup>1</sup></a>:  [Hu, Weihua & Fey, Matthias & Zitnik, Marinka & Dong, Yuxiao & Ren, Hongyu & Liu, Bowen & Catasta, Michele & Leskovec, Jure. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs.  ](https://arxiv.org/abs/2005.00687v6)
  
 <a id="note2" href="#note2ref"><sup>2</sup></a>:  [Dwivedi, Vijay & Joshi, Chaitanya & Laurent, Thomas & Bengio, Yoshua & Bresson, Xavier. (2020). Benchmarking Graph Neural Networks. ](https://arxiv.org/abs/2003.00982)