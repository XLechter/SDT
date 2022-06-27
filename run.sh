#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --mode 0 --datapath /mnt/data1/zwx/completion3d/data/pcn --dataset PCN --batch_size 4 --workers 16 --nepoch 300 --model_name Model --num_points 2048 --log_env PCN --lr 1e-4 --loss CD --use_mean_feature 0
