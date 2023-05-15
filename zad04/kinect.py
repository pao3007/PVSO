import pandas as pd
from matplotlib import pyplot as plt
from pyntcloud import PyntCloud
from freenect2 import Device, FrameType
import numpy as np
import open3d as o3d

def use_o3d(pts, write_text):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(pts)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    # read ply file
    pcd = o3d.io.read_point_cloud('my_pts.ply')

    # visualize
    o3d.visualization.draw_geometries([pcd])

#writeText = False

#pcd = o3d.io.read_point_cloud("/home/d618/PycharmProjects/CservenkaLukac/office/office1.pcd")

#o3d.io.write_point_cloud("/home/d618/PycharmProjects/CservenkaLukac/testModel.ply", pcd, write_ascii=writeText)

testPointCloud = o3d.io.read_point_cloud('testModel.ply')
#o3d.visualization.draw_geometries([testPointCloud])
testPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
plane_model, inliers = testPointCloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

inlier_cloud = testPointCloud.select_by_index(inliers)
outlier_cloud = testPointCloud.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

labels = np.array(testPointCloud.cluster_dbscan(eps=0.05, min_points=10))

segment_models={}
segments={}

max_plane_idx=20

rest=testPointCloud
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.05,ransac_n=3,num_iterations=100)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
