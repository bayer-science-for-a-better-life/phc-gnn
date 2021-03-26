#!/bin/bash
# python script call for 100K model

 python train_cifar10.py --input_embed_dim 224 --mp_units 224,224,224,224\
   --dropout_mpnn 0.1,0.1,0.1,0.1 --mp_norm naive-batch-norm --d_units 256,128\
   --aggr_msg mean --aggr_node mean --dropout_dn 0.2,0.1 --phm_dim 4 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir default-phm4 --n_runs 4 --epochs 1000\
   --w_init phm --c_init standard\
   --full_encoder True --nworkers 0 --device 0 --lr 0.001 --patience 10 --factor 0.5 --mlp_mp False\
   --weightdecay 0.01 --msg_encoder identity\
   --grad_clipping 2.0 --sc_type last