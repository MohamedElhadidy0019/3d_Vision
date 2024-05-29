import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_masks_folder(base_path):
    """Visualizes masks overlaid on the original RGB images in each folder."""
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        image_path = os.path.join(folder_path, 'rgb.jpg')
        txt_file_path = os.path.join(folder_path, 'mask.txt')
        
        # Read the image
        image = plt.imread(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        # Read and overlay the masks
        with open(txt_file_path, 'r') as f:
            for line in f:
                line_parts = line.strip().split(' ')
                class_id = int(line_parts[0])
                polygon = list(map(float, line_parts[1:]))

                # Convert normalized coordinates to image coordinates
                polygon = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0])) for i in range(0, len(polygon), 2)]

                # Create a polygon patch
                polygon_patch = patches.Polygon(polygon, edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(polygon_patch)
        
        plt.title(f'Masks for {folder}')
        plt.show()

# Example usage
base_path = 'dl_challenge/'
visualize_masks_folder(base_path)
