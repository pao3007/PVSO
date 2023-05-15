import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pcd = o3d.io.read_point_cloud("TLS_kitchen.ply")

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([1, 0, 0])

#o3d.io.write_point_cloud("office_test.ply", pcd, write_ascii=True)

#pcd = o3d.io.read_point_cloud('office_test.ply')

points = np.asarray(pcd.points)
scaler = StandardScaler()
normalize_points = scaler.fit_transform(points)
pca = PCA(n_components=3)
reduced_points = pca.fit_transform(normalize_points)

#wcss = []

#for i in range(1,11):
 #   kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=0)
  #  kmeans.fit(reduced_points)
  #  wcss.append(kmeans.inertia_)

num_clusters = 20#np.argmin(np.diff(wcss)) + 1
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(reduced_points)

labels = kmeans.predict(reduced_points)

colors = np.random.rand(num_clusters,3)

colored_labels = [colors[label] for label in labels]

pcd.colors = o3d.utility.Vector3dVector(colored_labels)

o3d.visualization.draw_geometries([pcd])



