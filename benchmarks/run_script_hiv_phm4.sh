#!/bin/bash

python train_hiv.py --input_embed_dim 200 --mp_units 200,200\
 --dropout_mpnn 0.2,0.2 --mp_norm naive-batch-norm --d_units 128,32\
 --aggr_msg softmax --aggr_node softmax --dropout_dn 0.3,0.1 --phm_dim 4 --learn_phm True --d_bn naive-batch-norm\
 --type undirectional-phm-sc-add --save_dir default-phm4 --n_runs 10 --epochs 50\
 --w_init phm --c_init standard\
 --full_encoder True --nworkers 0 --device 0 --lr 0.001 --patience 5 --factor 0.75 --mlp_mp True --weightdecay 0.1\
 --grad_clipping 2.0