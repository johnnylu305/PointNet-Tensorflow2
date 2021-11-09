import open3d as o3d
import numpy as np
import os
import glob


def get_data(args):
    path = args.dataset_dir
    path = os.path.join(path, "seg")
    saturn_color_map = {(0, 0, 1):0, (0, 1, 0):1}
    if args.phase=="train":
        # load training set
        print("Load training set")
        paths = glob.glob(os.path.join(path, "train", '*.ply'))
        train_size = len(paths)
        train_x = np.zeros((train_size, args.sampling, 3))
        train_y = np.zeros((train_size, 1))
        train_seg = np.zeros((train_size, args.sampling))
        for i, p in enumerate(paths):
            print("Load {}".format(p))
            point_cloud = load_point_cloud(p)
            # sample points
            point_cloud_np = np.asarray(point_cloud.points)
            idx = np.random.choice(point_cloud_np.shape[0], size=args.sampling, replace=False)
            point_cloud_np = point_cloud_np[idx]
            # zero-mean and normalized into an unit sphere.
            centroid = np.mean(point_cloud_np, axis=0)
            point_cloud_np = point_cloud_np - centroid
            train_x[i] = point_cloud_np/np.max(np.sqrt(np.sum(point_cloud_np**2, axis=1)))
            # color to label
            for j, color in enumerate(np.array(point_cloud.colors)[idx]):
                train_seg[i][j] = saturn_color_map[tuple(color)]

        # load validation set
        print("Load validation set")
        paths = glob.glob(os.path.join(path, "val", '*.ply'))
        val_size = len(paths)
        val_x = np.zeros((val_size, args.sampling, 3))
        val_y = np.zeros((val_size, 1))
        val_seg = np.zeros((val_size, args.sampling))
        for i, p in enumerate(paths):
            print("Load {}".format(p))
            point_cloud = load_point_cloud(p)
            # sample points
            point_cloud_np = np.asarray(point_cloud.points)
            idx = np.random.choice(point_cloud_np.shape[0], size=args.sampling, replace=False)
            point_cloud_np = point_cloud_np[idx]
            # zero-mean and normalized into an unit sphere.
            centroid = np.mean(point_cloud_np, axis=0)
            point_cloud_np = point_cloud_np - centroid
            val_x[i] = point_cloud_np/np.max(np.sqrt(np.sum(point_cloud_np**2, axis=1)))
            # color to label
            for j, color in enumerate(np.array(point_cloud.colors)[idx]):
                val_seg[i][j] = saturn_color_map[tuple(color)]       
            
        return train_x, train_y, train_seg, val_x, val_y, val_seg
    elif args.phase=="test":   
        # load test set
        print("Load test set")
        paths = glob.glob(os.path.join(path, "test", '*.ply'))
        test_size = len(paths)
        test_x = np.zeros((test_size, 2048*2, 3))
        test_y = np.zeros((test_size, 1))
        test_seg = np.zeros((test_size, 2048*2))
        for i, p in enumerate(paths):
            print("Load {}".format(p))
            point_cloud = load_point_cloud(p)
            # sample points
            point_cloud_np = np.asarray(point_cloud.points)
            # zero-mean and normalized into an unit sphere.
            centroid = np.mean(point_cloud_np, axis=0)
            point_cloud_np = point_cloud_np - centroid
            test_x[i] = point_cloud_np/np.max(np.sqrt(np.sum(point_cloud_np**2, axis=1)))
            # color to label
            for j, color in enumerate(np.array(point_cloud.colors)):
                test_seg[i][j] = saturn_color_map[tuple(color)]       
        return test_x, test_y, test_seg


def load_point_cloud(path):
    pcd_load = o3d.io.read_point_cloud(path)
    return pcd_load


def shuffle(x, y, seg):
    dataset_size = len(x)
    dataset_ids = np.arange(dataset_size)
    np.random.shuffle(dataset_ids)
    return x[dataset_ids], y[dataset_ids], seg[dataset_ids]
