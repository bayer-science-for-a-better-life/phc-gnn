#!/bin/bash

 python train_pcba.py --input_embed_dim 512 --mp_units 512,512,512,512,512,512,512\
   --dropout_mpnn 0.1,0.1,0.1,0.1,0.1,0.1,0.1 --mp_norm naive-batch-norm --d_units 768,256\
   --aggr_msg sum --aggr_node sum --dropout_dn 0.3,0.2 --phm_dim 4 --learn_phm True --d_bn naive-batch-norm\
   --type undirectional-phm-sc-add --save_dir default-phm4 --n_runs 5 --epochs 150\
   --w_init phm --c_init standard\
   --full_encoder True --nworkers 0 --device 0 --lr 0.0005 --patience 5 --factor 0.75 --mlp_mp False\
   --weightdecay 0.0001 --msg_encoder identity\
   --grad_clipping 2.0