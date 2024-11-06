#!/bin/bash
CKP=$1
DATA=$2
DEN=$3
NPT=$4
STEP=$5
python GMSF2D/main_gmsf.py \
    --checkpoint_dir $CKP \
    --stage particle \
    --root '/home/data/particle-tracking/GMSF/' \
    --dataset $DATA \
    --snr 7 \
    --density $DEN \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints $NPT \
    --lr 2e-4 --batch_size 64 --num_steps $STEP
    