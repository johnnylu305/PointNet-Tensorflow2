import open3d as o3d
import numpy as np
import os
import glob


def get_data(args):
    # (0, 0, 1), 0, sphere
    # (0, 1, 0), 1, torus
    saturn_color_map = {(0, 0, 1):0, (0, 1, 0):1}
    path = args.dataset_dir
    path = os.path.join(path, "cla")
    if args.phase=="train":
        # load training set
        print("Load training set")
        paths = glob.glob(os.path.join(path, "train", '*.ply'))
        train_size = len(paths)
        train_x = np.zeros((train_size, args.sampling, 3))
        train_y = np.zeros((train_size, 2))
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
            for color in np.unique(np.array(point_cloud.colors), axis=0):
                train_y[i, saturn_color_map[tuple(color)]] = 1
        # load validation set
        print("Load validation set")
        paths = glob.glob(os.path.join(path, "val", '*.ply'))
        val_size = len(paths)
        val_x = np.zeros((val_size, args.sampling, 3))
        val_y = np.zeros((val_size, 2))
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
            for color in np.unique(np.array(point_cloud.colors), axis=0):
                val_y[i, saturn_color_map[tuple(color)]] = 1
            
        return train_x, train_y, val_x, val_y
    elif args.phase=="test":   
        # load testing set
        print("Load test set")
        paths = glob.glob(os.path.join(path, "test", '*.ply'))
        test_size = len(paths)
        test_x = np.zeros((test_size, args.sampling, 3))
        test_y = np.zeros((test_size, 2))
        test_name = []
        for i, p in enumerate(paths):
            print("Load {}".format(p))
            point_cloud = load_point_cloud(p)
            test_name.append(p.split("/")[-1])
            # sample points
            point_cloud_np = np.asarray(point_cloud.points)
            idx = np.random.choice(point_cloud_np.shape[0], size=args.sampling, replace=False)
            point_cloud_np = point_cloud_np[idx]
            # zero-mean and normalized into an unit sphere.
            centroid = np.mean(point_cloud_np, axis=0)
            point_cloud_np = point_cloud_np - centroid
            test_x[i] = point_cloud_np/np.max(np.sqrt(np.sum(point_cloud_np**2, axis=1)))
            for color in np.unique(np.array(point_cloud.colors), axis=0):
                test_y[i, saturn_color_map[tuple(color)]] = 1
        return test_x, test_y, test_name


def load_point_cloud(path):
    pcd_load = o3d.io.read_point_cloud(path)
    return pcd_load


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
