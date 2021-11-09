import numpy as np
import open3d as o3d
import time

def generate_abstract(means, variances, category, size, color_map):
    cov = np.eye(3)
    cov[(0, 1, 2), (0, 1, 2)] = variances
    data = np.random.multivariate_normal(means, cov, size)
    label = np.full(size, category)
    label_color = np.full((size, 3), color_map)
    return data, label, label_color


def generate_sphere(r, category, size, color_map):
    u = np.random.random(size)*2*np.pi
    v = np.random.random(size)*2*np.pi
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    label = np.full(size, category)
    label_color = np.full((size, 3), color_map)
    return np.vstack((x, y, z)).T, label, label_color


def generate_torus(r, R, category, size, color_map):
    u = np.random.random(size)*2*np.pi
    v = np.random.random(size)*2*np.pi
    x = (R+r*np.cos(u))*np.cos(v)
    y = (R+r*np.cos(u))*np.sin(v)
    z = r*np.sin(u)
    label = np.full(size, category)
    label_color = np.full((size, 3), color_map)
    return np.vstack((x, y, z)).T, label, label_color


def transform(target, rotate, trans, scale):
    m = np.eye(4)*scale
    target = np.concatenate((target, np.ones((target.shape[0], 1))), axis=1)
    m[(0, 1, 2), (3, 3, 3)] = trans
    rX, rY, rZ = np.eye(4), np.eye(4), np.eye(4)
    a, b, c = rotate
    rX[(1, 1, 2, 2), (1, 2, 1, 2)] = [np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]
    rY[(0, 0, 2, 2), (0, 2, 0, 2)] = [np.cos(b), np.sin(b), -np.sin(b), np.cos(b)]
    rZ[(0, 0, 1, 1), (0, 1, 0, 1)] = [np.cos(c), -np.sin(c), np.sin(c), np.cos(c)]
    m = m@rX@rY@rZ
    # this can be (a@b.T).T = a.T@b
    return (m@target.T).T[:, :3]


def generateSaturn(n, path):
    name = "saturn_"
    for i in range(0, n, 3):
        # get sphere
        # 0.2~2.0
        scale1 = 0.2+np.random.random()*1.8
        radius1 = 1*scale1
        data1, label1, color1 = generate_sphere(radius1, 0, 2048, [0, 0, 1])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data1)
        pcd.colors = o3d.utility.Vector3dVector(color1)
        # save
        o3d.io.write_point_cloud("{}/{}{}.ply".format(path, 'sphere_', str(i)), pcd)

        # save poincloud as png
        vis = o3d.visualization.Visualizer()
        # you can set visible to False in some systems
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("{}/{}{}.png".format(path, 'sphere_', str(i)))
        vis.destroy_window()

        # get torus
        # 0.2~1.0
        scale2 = 0.2+np.random.random()*0.8
        r = scale2*0.3
        R = radius1+scale2*4
        data2, label2, color2 = generate_torus(r, R, 1, 2048, [0, 1, 0])
        
        data2_trans = transform(data2, [0.55*np.pi, 0.2*np.pi, 0.1*np.pi], [0, 0, 5], scale2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data2_trans)
        pcd.colors = o3d.utility.Vector3dVector(color2)
        # save
        o3d.io.write_point_cloud("{}/{}{}.ply".format(path, 'torus_', str(i+1)), pcd)

        # save poincloud as png
        vis = o3d.visualization.Visualizer()
        # you can set visible to False in some systems
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("{}/{}{}.png".format(path, 'torus_', str(i+1)))
        vis.destroy_window()

        # form saturn
        saturn = np.vstack((data1, data2))
        saturn = transform(saturn, [0.55*np.pi, 0.2*np.pi, 0.1*np.pi], [0, 0, 5], scale2)
        color = np.vstack((color1, color2))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(saturn)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # save
        o3d.io.write_point_cloud("{}/{}{}.ply".format(path, 'saturn_', str(i+2)), pcd)
        # load and plot
        #pcd_load = o3d.io.read_point_cloud("./test/sync.ply")
        #o3d.visualization.draw_geometries([pcd_load])

        # save poincloud as png
        vis = o3d.visualization.Visualizer()
        # you can set visible to False in some systems
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_image("{}/{}{}.png".format(path, 'saturn_', str(i+2)))
        vis.destroy_window()

if __name__=="__main__":
    generateSaturn(2700, "./train")
    generateSaturn(900, "./val")
    generateSaturn(900, "./test")
    
