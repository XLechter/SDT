import os

import argparse
from test import test
from train.train import train
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Completion')

    # mode and dataset
    parser.add_argument('--mode', type=int, default=0, help='0 for train, 1 for test')
    parser.add_argument('--model_dir', type=str, default='/mnt/data2/zwx/ECG/log/PCT_CD_train/PCT_4SA_2021-02-26T16:10:21') # for test only
    parser.add_argument('--dataset', type=str, default='Completion3D', help='dataset')
    parser.add_argument('--datapath', type=str, default='data/completion3d', help='dataset path')
    # common args
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='ECG',  help='model to use')
    parser.add_argument('--load_model', type=str, default='',  help='load model to resume training / start testing')
    parser.add_argument('--resume_epoch', type=int, default=0, help='which epoch to resume from')
    parser.add_argument('--num_points', type=int, default=2048,  help='number of ground truth points')
    parser.add_argument('--log_env', type=str, default="ecg_2048", help='subfolder name inside log/<model>_<loss>_train/')
    parser.add_argument('--loss', type=str, default='EMD', help='train loss type; CD or EMD')
    parser.add_argument('--manual_seed', type=str, default='', help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # cascade, msn, pcn:0.0001, topnet:0.5e-2
    parser.add_argument('--use_mean_feature', type=int, default=0, help='0 if not using, 1 if using')

    args = parser.parse_args()

    #assert args.model_name in list(models_dict.keys())
    assert args.loss == 'EMD' or args.loss == 'CD'

    if args.mode == 0:
        print('args.num_points in train', args.num_points)
        train(args)
    else:
        test(args)










