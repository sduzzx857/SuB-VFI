#!/bin/bash
PORT=$1
CFG=$2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $PORT train.py --config $CFG