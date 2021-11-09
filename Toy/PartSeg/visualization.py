import open3d as o3d
import numpy as np
import os
import tensorflow as tf
from mapping import catidToName, colorMap 

class Vis:
    def __init__(self, path):
        self.cat_seen_count = np.zeros(1)       
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def visualize(self, xs, cls_labels, preds, n=1):
        for x, cls_label, pred in zip(xs[:n], cls_labels[:n], preds[:n]):
            x = x[:pred.shape[0]]
            cls_label = int(cls_label[0])
            name = catidToName[cls_label]
            
            save_path = os.path.join(self.path, "{}_{}.ply".format(name, int(self.cat_seen_count[cls_label])))
            self.cat_seen_count[cls_label] += 1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x)
            color = colorMap[pred]
            pcd.colors = o3d.utility.Vector3dVector(color)
            # save
            o3d.io.write_point_cloud(save_path, pcd)
            
            # save poincloud as png
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(pcd)
            # vis.update_geometry(pcd)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("{}.png".format(save_path[:-4]))
            vis.destroy_window()
