#!/bin/bash

python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/VESICLE7high1100_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'VESICLE' \
    --snr 7 \
    --density high \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 1100 \
    --eval

python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/VESICLE7low135_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'VESICLE' \
    --snr 7 \
    --density low \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 135 \
    --eval

python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/MICROTUBULE7high750_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'MICROTUBULE' \
    --snr 7 \
    --density high \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 750 \
    --eval

python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/MICROTUBULE7low85_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'MICROTUBULE' \
    --snr 7 \
    --density low \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 85 \
    --eval


python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/RECEPTOR7high910_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'RECEPTOR' \
    --snr 7 \
    --density high \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 910 \
    --eval

python GMSF2D/main_gmsf.py --resume GMSF2D/pretrained/RECEPTOR7low110_64_4.pth \
    --stage particle \
    --root 'data/GMSF/' \
    --dataset 'RECEPTOR' \
    --snr 7 \
    --density low \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 4 \
    --feature_channels 64 \
    --npoints 110 \
    --eval