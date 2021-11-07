import numpy as np
import os
import glob
import h5py
import json


def get_data(args):
    path = args.dataset_dir
    if args.phase=="train":
        # load training set
        print("Load training set")
        path = os.path.join(path, "data", "shapenet_part_seg_hdf5_data", "hdf5_data")
        with open(os.path.join(path, "train_hdf5_file_list.txt")) as r:
            files = r.readlines()
        size = 2048*5+1897
        train_x = np.zeros((size, 2048, 3))
        train_y = np.zeros((size, 1))
        train_seg = np.zeros((size, 2048))
        start = 0
        for file in files:
            # each point cloud has been normalized
            x, y, seg = load_point_clouds(os.path.join(path, file.rstrip()))
            end = start+len(x)
            train_x[start:end] = x
            train_y[start:end] = y
            train_seg[start:end] = seg
            start = end
        
        # load validation set
        print("Load validation set")
        with open(os.path.join(path, "val_hdf5_file_list.txt")) as r:
            files = r.readlines()
        size = 1870
        val_x = np.zeros((size, 2048, 3))
        val_y = np.zeros((size, 1))
        val_seg = np.zeros((size, 2048))
        start = 0
        for file in files:
            # each point cloud has been normalized
            x, y, seg = load_point_clouds(os.path.join(path, file.rstrip()))
            end = start+len(x)
            val_x[start:end] = x
            val_y[start:end] = y
            val_seg[start:end] = seg
            start = end       
        return train_x, train_y, train_seg, val_x, val_y, val_seg
    elif args.phase=="test":   
        # load test set
        print("Load test set")
        path = os.path.join(path, "data", "shapenet_part_seg_hdf5_data", "hdf5_data")
        with open(os.path.join(path, "test_hdf5_file_list.txt")) as r:
            files = r.readlines()
        size = 2048+826
        test_x = np.zeros((size, 2048, 3))
        test_y = np.zeros((size, 1))
        test_seg = np.zeros((size, 2048))
        start = 0
        for file in files:
            # each point cloud has been normalized
            x, y, seg = load_point_clouds(os.path.join(path, file.rstrip()))
            end = start+len(x)
            test_x[start:end] = x
            test_y[start:end] = y
            test_seg[start:end] = seg
            start = end
        return test_x, test_y, test_seg


def load_point_clouds(path):
    pcd_load = h5py.File(path)
    return pcd_load['data'], pcd_load['label'], pcd_load['pid']


def shuffle(x, y, seg):
    dataset_size = len(x)
    dataset_ids = np.arange(dataset_size)
    np.random.shuffle(dataset_ids)
    return x[dataset_ids], y[dataset_ids], seg[dataset_ids]
