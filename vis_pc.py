import open3d as o3d
import numpy as np
import os





def main():
    pcd_path = \
        'dl_challenge/001/pc.npy'

    # Load the point cloud from the npy file
    point_cloud = np.load(pcd_path)
    print(f'point_cloud.shape = {point_cloud.shape}')

    # Reshape the point cloud array to (N, 3) format
    point_cloud = np.transpose(point_cloud, (1, 2, 0)).reshape(-1, 3)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    main()