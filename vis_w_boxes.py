import open3d as o3d
import numpy as np

def visualize_point_cloud_and_bounding_boxes(point_cloud_file, bounding_boxes_file):
    # Load the point cloud from the npy file
    point_cloud = np.load(point_cloud_file)
    # Reshape the point cloud array to (N, 3) format
    point_cloud = np.transpose(point_cloud, (1, 2, 0)).reshape(-1, 3)
    # only leave points of z between 0 and 1.5
    point_cloud = point_cloud[(point_cloud[:, 2] > 0) & (point_cloud[:, 2] < 1.5)]
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)



    # Load the bounding boxes from the npy file
    bounding_boxes = np.load(bounding_boxes_file)
    
    # Function to create lines for the bounding boxes
    def create_bbox_lines(bounding_boxes):
        lines = []
        line_points = []
        point_index = 0
        for bbox_vertices in bounding_boxes:
            # Extract vertices of the bounding box
            vertices = bbox_vertices[:8]
            line_points.extend(vertices)
            # Define lines from the vertices
            lines += [
                [point_index, point_index+1],
                [point_index+1, point_index+2],
                [point_index+2, point_index+3],
                [point_index+3, point_index],
                
                [point_index+4, point_index+5],
                [point_index+5, point_index+6],
                [point_index+6, point_index+7],
                [point_index+7, point_index+4],
                
                [point_index, point_index+4],
                [point_index+1, point_index+5],
                [point_index+2, point_index+6],
                [point_index+3, point_index+7]
            ]
            point_index += 8
        return line_points, lines

    # Get the points and lines for the bounding boxes
    bbox_points, bbox_lines = create_bbox_lines(bounding_boxes)
    
    # Create Open3D LineSet for bounding boxes
    bbox_lines_set = o3d.geometry.LineSet()
    bbox_lines_set.points = o3d.utility.Vector3dVector(bbox_points)
    bbox_lines_set.lines = o3d.utility.Vector2iVector(bbox_lines)


    # make the view start from flipped z
    # flip the view
    # look at the point cloud from the top
    ctr = o3d.visualization.Visualizer()
    ctr.create_window()
    ctr.add_geometry(pcd)
    ctr.add_geometry(bbox_lines_set)
    ctr.get_view_control().set_front([0, 0, -1])
    ctr.get_view_control().set_up([0, -1, 0])
    ctr.run()
    ctr.destroy_window()
    # Visualize the point cloud and bounding boxes
    # o3d.visualization.draw_geometries([pcd, bbox_lines_set])


def main():

    for i in range(0, 199):
        # i=1
        point_cloud_file = f'dl_challenge/{i:03d}/pc.npy'
        bounding_boxes_file = f'dl_challenge/{i:03d}/bbox3d.npy'
        print(f'currently showing number {i:03d}')
        visualize_point_cloud_and_bounding_boxes(point_cloud_file, bounding_boxes_file)
        # break

if __name__ == "__main__":
    main()
