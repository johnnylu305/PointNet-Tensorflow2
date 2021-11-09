import open3d as o3d
import numpy as np
import os
import tensorflow as tf

def visualize(xs, labels, preds, names, path, n=1):
    if not os.path.exists(path):
        os.makedirs(path)
    category_map = {(0, 1):'torus', (1, 0):'sphere', (1, 1):'saturn'}
    for x, name, pred in zip(xs[:n], names[:n], preds[:n]):
        pred = tuple(tf.where(pred>=0.5, 1, 0).numpy())
        save_path = os.path.join(path, "{}_{}".format(category_map[pred], name.split("_")[-1]))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        #pcd.colors = o3d.utility.Vector3dVector(color)
        # save
        o3d.io.write_point_cloud(save_path, pcd)
        
        # save poincloud as png
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.update_geometry()
        #vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("{}.png".format(save_path[:-4]))
        vis.destroy_window()
