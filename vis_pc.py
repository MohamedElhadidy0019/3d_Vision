import open3d as o3d
import numpy as np






def main():
    pcd_path = \
        'dl_challenge/8b061a8b-9915-11ee-9103-bbb8eae05561/pc.npy'
    # pcd_npy = np.load(pcd_path)
    # print(pcd_npy.shape)
    # pcd_npy = pcd_npy.transpose((1, 2, 0))  # (height, width, channels) -> (height, width, 3)
    # print(pcd_npy.shape)
    # print(f'type = {pcd_npy.dtype}')
    # # type to float 32
    # pcd_npy = pcd_npy.astype(np.float32)

    # # Create an Open3D PointCloud object
    # point_cloud = o3d.geometry.PointCloud()

    # # Convert the NumPy array to Open3D Vector3dVector format
    # point_cloud.points = o3d.utility.Vector3dVector(pcd_npy)

    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([point_cloud])

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