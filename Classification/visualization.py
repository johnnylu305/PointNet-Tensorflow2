import open3d as o3d
import numpy as np
import os
import tensorflow as tf

def visualize(xs, labels, preds, names, path, n=1):
    if not os.path.exists(path):
        os.makedirs(path)
    category_map = {0:'airplane', 1:'bathtub', 2:'bed', 3:'bench', 4:'bookshelf', 
                    5:'bottle', 6:'bowl', 7:'car', 8:'chair', 9:'cone',
                    10:'cup', 11:'curtain', 12:'desk', 13:'door', 14:'dresser',
                    15:'flower_pot', 16:'glass_box', 17:'guitar', 18:'keyboard', 19:'lamp',
                    20:'laptop', 21:'mantel', 22:'monitor', 23:'night_stand', 24:'person',
                    25:'piano', 26:'plant', 27:'radio', 28:'range_hood', 29:'sink', 
                    30:'sofa', 31:'stairs', 32:'stool', 33:'table', 34:'tent',
                    35:'toilet' , 36:'tv_stand' , 37:'vase' ,38:'wardrobe', 39:'xbox'}
    for x, name, pred in zip(xs[:n], names[:n], preds[:n]):
        name = name[0].split("/")[-1][:-4]
        # label_pred
        save_path = os.path.join(path, "{}_{}.ply".format(name, category_map[np.argmax(pred)]))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        # save
        o3d.io.write_point_cloud(save_path, pcd)
        
        # save poincloud as png
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("{}.png".format(save_path[:-4]))
        vis.destroy_window()
