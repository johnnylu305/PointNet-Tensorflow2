import numpy as np
import os
import glob
import h5py
import json
from mapping import strToCatid, catidToName, nameToParts


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

# Reference: charlesq34/pointnet
def get_org_data(args):
    path = args.dataset_dir
    # load test set
    print("Load test set")
    path = os.path.join(path, "data")
    # number of shape
    size = 2048+826
    test_x = np.zeros((size, args.max_size, 3))
    test_y = np.zeros((size, 1))
    test_seg = []
    test_size = np.zeros((size, 1))
    with open(os.path.join(path, "testing_ply_file_list.txt")) as r:
        for i, f in enumerate(r):
            x_path, seg_label_path, cls_label = f.split(" ")
            x_path = os.path.join(path, "shapenetcore_partanno_v0", "PartAnnotation", x_path)
            seg_label_path = os.path.join(path, "shapenetcore_partanno_v0", "PartAnnotation", seg_label_path)
            # get class label
            test_y[i][0] = int(strToCatid[cls_label.rstrip()])
            # get point cloud
            with open(x_path, 'r') as rr:
                pts_str = [item.rstrip() for item in rr.readlines()]
                pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
            # zero-mean and normalized into an unit sphere.
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            pts = pts/np.max(np.sqrt(np.sum(pts**2, axis=1)))
            # get # of points
            test_size[i][0] = pts.shape[0]
            # padding
            while pts.shape[0]<args.max_size:
                pts = np.concatenate((pts, pts), axis=0)
            test_x[i] = pts[:args.max_size]
            # get segmentation label
            with open(seg_label_path, 'r') as rr:
                part = np.array([int(item.rstrip()) for item in rr.readlines()], dtype=np.uint8)
                test_seg.append(np.array([nameToParts[catidToName[test_y[i][0]]][x-1] for x in part])[:args.max_size])
    return test_x, test_y, test_seg, test_size


def load_point_clouds(path):
    pcd_load = h5py.File(path)
    return pcd_load['data'], pcd_load['label'], pcd_load['pid']


def shuffle(x, y, seg):
    dataset_size = len(x)
    dataset_ids = np.arange(dataset_size)
    np.random.shuffle(dataset_ids)
    return x[dataset_ids], y[dataset_ids], seg[dataset_ids]
