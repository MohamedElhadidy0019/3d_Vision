import open3d







def main():
    pcd = open3d.io.read_point_cloud("vis_pc.ply")
    open3d.visualization.draw_geometries([pcd])










if __name__ == "__main__":
    main()