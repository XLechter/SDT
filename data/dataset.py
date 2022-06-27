import torch
import numpy as np
import torch.utils.data as data
import h5py
import os

def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()

    #return cloud_data.astype(np.float64)
    return cloud_data

def pad_cloudN(P, Nin):
    """ Pad or subsample 3D Point cloud to Nin number of points """
    N = P.shape[0]
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__
    choice = np.arange(N)
    if N > Nin: # need to subsample
        ii = rs.choice(N, Nin)
        choice = ii
    elif N < Nin: # need to pad by duplication
        ii = rs.choice(N, Nin - N)
        choice = np.concatenate([range(N),ii])
    P = P[choice, :]

    return P

class Completion3D(data.Dataset):
    def __init__(self, datapath, train=True, npoints=2048, use_mean_feature=0, benchmark=False):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints
        self.train = train
        self.use_mean_feature = use_mean_feature
        if train:
            split = 'train'
        elif benchmark:
            split = 'test'
        else:
            split = 'val'

        DATA_PATH = datapath

        self.partial_data_paths = [os.path.join(DATA_PATH, split, 'partial', k.rstrip()+ '.h5') for k in open(DATA_PATH + '/%s.list' % (split)).readlines()]  #sorted() 

        if benchmark:
            self.gt_data_paths = self.partial_data_paths
        else:
            self.gt_data_paths = [os.path.join(DATA_PATH, split, 'gt', k.rstrip() + '.h5') for k in
                                         open(DATA_PATH + '/%s.list' % (split)).readlines()]  #sorted()
        #print(self.partial_data_paths, np.array(self.partial_data_paths).shape)
        self.len = np.array(self.partial_data_paths).shape[0]
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(np.array(load_h5(self.partial_data_paths[index]))).float()
        #print('partial.shape', partial.shape)
        complete = torch.from_numpy(np.array(load_h5(self.gt_data_paths[index]))).float()
        label = self.partial_data_paths[index]
        if self.use_mean_feature == 1:
            mean_feature_input = torch.from_numpy(np.array(self.mean_feature[label])).float()
            return label, partial, complete, mean_feature_input
        else:
            return label, partial, complete

class PCN(data.Dataset):
    def __init__(self, datapath, train=True, npoints=2048, use_mean_feature=0, test=False):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints
        self.train = train
        self.use_mean_feature = use_mean_feature
        if train:
            split = 'train'
        elif test:
            split = 'test'
        else:
            split = 'val'

        DATA_PATH = datapath

        self.partial_data_paths = [os.path.join(DATA_PATH, split, 'partial', k.rstrip()+ '.h5') for k in open(DATA_PATH + '/%s.list' % (split)).readlines()] #sorted()

        self.gt_data_paths = [os.path.join(DATA_PATH, split, 'gt', k.rstrip() + '.h5') for k in
                                     open(DATA_PATH + '/%s.list' % (split)).readlines()]  #sorted()
        #print(self.partial_data_paths, np.array(self.partial_data_paths).shape)
        self.len = np.array(self.partial_data_paths).shape[0]
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(pad_cloudN(np.array(load_h5(self.partial_data_paths[index])), 2048)).float()
        #print('partial.shape', partial.shape)
        complete = torch.from_numpy(np.array(load_h5(self.gt_data_paths[index]))).float()
        label = self.partial_data_paths[index]
        if self.use_mean_feature == 1:
            mean_feature_input = torch.from_numpy(np.array(self.mean_feature[label])).float()
            return label, partial, complete, mean_feature_input
        else:
            return label, partial, complete


class SCAN(data.Dataset):
    def __init__(self, datapath, npoints=2048):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints

        DATA_PATH = datapath

        self.partial_data_paths = [os.path.join(DATA_PATH, k.rstrip() + '.h5') for k in open(DATA_PATH + '/data_list.txt').readlines()] #sorted()

        self.gt_data_paths = self.partial_data_paths

        self.len = np.array(self.partial_data_paths).shape[0]
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(pad_cloudN(np.array(load_h5(self.partial_data_paths[index])), 2048)).float()
        #print('partial.shape', partial.shape)
        complete = torch.from_numpy(np.array(load_h5(self.gt_data_paths[index]))).float()
        label = self.partial_data_paths[index]

        return label, partial, complete


class KITTI(data.Dataset):
    def __init__(self, datapath, npoints=2048):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints

        DATA_PATH = datapath

        self.partial_data_paths = [os.path.join(DATA_PATH, 'cars_h5', k.rstrip() + '.h5') for k in open(DATA_PATH + '/data_list.txt').readlines()] #sorted()

        self.gt_data_paths = self.partial_data_paths

        self.len = np.array(self.partial_data_paths).shape[0]
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(pad_cloudN(np.array(load_h5(self.partial_data_paths[index])), 2048)).float()
        complete = torch.from_numpy(np.array(load_h5(self.gt_data_paths[index]))).float()
        label = self.partial_data_paths[index]

        return label, partial, complete
