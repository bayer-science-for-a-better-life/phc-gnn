#!/bin/bash
# python script call for 100K model

 python train_zinc.py --input_embed_dim 180 --mp_units 180,180,180,180\
   --dropout_mpnn 0,0,0,0 --mp_norm naive-batch-norm --d_units 180,80\
   --aggr_msg sum --aggr_node sum --dropout_dn 0.2,0.1 --phm_dim 4 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir default-phm4 --n_runs 4 --epochs 1000\
   --w_init phm --c_init standard\
   --full_encoder True --nworkers 0 --device 0 --lr 0.001 --patience 10 --factor 0.5 --mlp_mp True\
   --weightdecay 0.01 --msg_encoder identity\
   --grad_clipping 2.0 --sc_type last