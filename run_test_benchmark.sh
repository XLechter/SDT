#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_benchmark.py --datapath /mnt/data2/zwx/completion3d/data/shapenet --model_dir /mnt/data2/zwx/ECG/trained_model --mode 1 --dataset Completion3D --batch_size 1 --num_points 16384 --model_name Model --log_env PCN --lr 0.0001 --loss CD --use_mean_feature 0 --workers 16 --nepoch 300
