import open3d as o3d
import numpy as np

# Generate some random data
np.random.seed(0)
n_samples = 100
points = np.random.rand(n_samples, 3) * 10
points[:20, 2] += 50  # Add some outliers to the data

# Define a plane model for RANSAC
plane_model, inliers = None, None
threshold = 0.3

# Run RANSAC algorithm
for i in range(100):
    # Randomly select three points
    indices = np.random.choice(n_samples, 3, replace=False)
    selected_points = points[indices]

    # Fit a plane to the selected points
    plane_model, inliers = o3d.geometry.(
        o3d.geometry.PointCloud(points),
        distance_threshold=threshold,
        plane_initial=selected_points)

    # Check if the model fits enough inliers
    if len(inliers) > n_samples * 0.7:
        break

# Remove outliers from the data
outliers = np.delete(points, inliers, axis=0)

# Visualize the results
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
inlier_pcd = pcd.select_by_index(inliers)
outlier_pcd = pcd.select_by_index(outliers)
inlier_pcd.paint_uniform_color([0, 1, 0])
outlier_pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])