import numpy as np
import open3d as o3d
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pcd = o3d.io.read_point_cloud("TLS_kitchen.ply")

points = np.asarray(pcd.points)
scaler = StandardScaler()
normalize_points = scaler.fit_transform(points)
pca = PCA(n_components=3)
reduced_points = pca.fit_transform(normalize_points)

birch = Birch(threshold=0.1, n_clusters=20)

birch.fit(reduced_points)

labels = birch.predict(reduced_points)

colors = np.random.rand(np.unique(labels).shape[0],3)

colored_labels = [colors[label] for label in labels]

pcd.colors = o3d.utility.Vector3dVector(colored_labels)

o3d.visualization.draw_geometries([pcd])