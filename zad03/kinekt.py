import pandas as pd
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

writeText = False
# Open the default device and capture a color and depth frame.
device = Device()
frames = {}
with device.running():
    for type_, frame in device:
        frames[type_] = frame
        if FrameType.Color in frames and FrameType.Depth in frames:
            break

# Use the factory calibration to undistort the depth frame and register the RGB
# frame onto it.
rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]
undistorted, registered, big_depth = device.registration.apply(
    rgb, depth, with_big_depth=True)

# Combine the depth and RGB data together into a single point cloud.
with open('output.pcd', 'wb') as fobj:
    device.registration.write_pcd(fobj, undistorted, registered)

with open('output_big.pcd', 'wb') as fobj:
   device.registration.write_big_pcd(fobj, big_depth, rgb)

pcd = o3d.io.read_point_cloud("/home/d618/Desktop/dump_pcd")

o3d.io.write_point_cloud("/home/d618/PycharmProjects/CservenkaLukac/my_pts.ply", pcd, write_ascii=writeText)

pcd = o3d.io.read_point_cloud('my_pts.ply')
o3d.visualization.draw_geometries([pcd])