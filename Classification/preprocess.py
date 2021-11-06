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
        with open(os.path.join(path, "data", "modelnet40_ply_hdf5_2048", "train_files.txt")) as r:
            files = r.readlines()
        size = 4*2048+1648
        train_x = np.zeros((size, 2048, 3))
        train_y = np.zeros((size, 1))
        start = 0
        for file in files:
            # each point cloud has been normalized
            x, y = load_point_clouds(os.path.join(path, file.rstrip()))
            end = start+len(x)
            train_x[start:end] = x
            train_y[start:end] = y
            start = end
            
        # load validation set
        print("Load validation set")
        with open(os.path.join(path, "data", "modelnet40_ply_hdf5_2048", "test_files.txt")) as r:
            files = r.readlines()
        size = 2048+420
        val_x = np.zeros((size, 2048, 3))
        val_y = np.zeros((size, 1))
        start = 0
        for file in files:
            # each point cloud has been normalized
            x, y = load_point_clouds(os.path.join(path, file.rstrip()))
            end = start+len(x)
            val_x[start:end] = x
            val_y[start:end] = y
            start = end
        return train_x, train_y, val_x, val_y
    elif args.phase=="test":   
        # load test set
        print("Load test set")
        with open(os.path.join(path, "data", "modelnet40_ply_hdf5_2048", "test_files.txt")) as r:
            files = r.readlines()
        size = 2048+420
        test_x = np.zeros((size, 2048, 3))
        test_y = np.zeros((size, 1))
        test_name = []
        start = 0
        for i, file in enumerate(files):
            # each point cloud has been normalized
            x, y, name = load_point_clouds(os.path.join(path, file.rstrip()), i)
            end = start+len(x)
            test_x[start:end] = x
            test_y[start:end] = y
            test_name += name
            start = end
        test_name = np.array(test_name).reshape(size, 1)
        return test_x, test_y, test_name


def sample(x, n=1024):
    # sample points
    idx = np.random.choice(x.shape[1], size=n, replace=False)
    x = x[:,idx,:]
    return x


def load_point_clouds(path, idx=None):
    pcd_load = h5py.File(path)
    if idx is not None:
        with open(os.path.join("{}_{}_id2file.json".format(path[:-4], idx))) as f:
            names = json.load(f)
        return pcd_load['data'], pcd_load['label'], names
    else:
        return pcd_load['data'], pcd_load['label']


def shuffle(x, y):
    dataset_size = len(x)
    dataset_ids = np.arange(dataset_size)
    np.random.shuffle(dataset_ids)
    return x[dataset_ids], y[dataset_ids]


def rotate_point_cloud_angle(x, angle):
    rotation_matrix = np.eye(3, dtype=np.float32)
    rotation_matrix[(0, 0, 2, 2), (0, 2, 0, 2)] = [np.cos(angle), 
                                                   np.sin(angle), 
                                                   -np.sin(angle), 
                                                   np.cos(angle)]
    return x@np.transpose(rotation_matrix[np.newaxis, :, :], [0, 2, 1])


def rotate_point_cloud(x):
    batch = x.shape[0]
    rotation_matrix = np.tile(np.eye(3, dtype=np.float32), (batch, 1, 1))
    angle = np.random.uniform(size=batch)*2*np.pi
    rotation_matrix[:, (0, 0, 2, 2), (0, 2, 0, 2)] = np.stack((np.cos(angle), 
                                                               np.sin(angle), 
                                                               -np.sin(angle), 
                                                               np.cos(angle)), axis=1)
    x = x@np.transpose(rotation_matrix, [0, 2, 1])
    return x


def jitter_point_cloud(x, sigma=0.01, clip=0.05):
    assert(clip>0)
    return x+np.clip(sigma*np.random.normal(size=x.shape), -1*clip, clip)
