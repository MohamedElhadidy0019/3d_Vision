import numpy as np
import os
import cv2

def mask_to_polygons(mask):
    """Convert a binary mask to polygon format using OpenCV."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten()
        if len(contour) >= 6:  # A valid polygon must have at least 3 points (6 coordinates)
            polygons.append(contour)
    return polygons

def normalize_coordinates(polygons, width, height):
    """Normalize polygon coordinates to be between 0 and 1."""
    normalized_polygons = []
    for polygon in polygons:
        normalized_polygon = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / width
            y = polygon[i + 1] / height
            normalized_polygon.append(x)
            normalized_polygon.append(y)
        normalized_polygons.append(normalized_polygon)
    return normalized_polygons

def create_yolo_files(base_path, class_id=0):
    """Creates YOLO annotation files with bounding boxes and masks."""
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        mask_path = os.path.join(folder_path, 'mask.npy')
        if not os.path.exists(mask_path):
            continue
        
        masks = np.load(mask_path)
        Nchannels, height, width = masks.shape
        txt_file = os.path.join(folder_path, 'mask.txt')

        with open(txt_file, 'w') as f:
            for channel in range(Nchannels):
                mask = masks[channel]
                polygons = mask_to_polygons(mask)
                normalized_polygons = normalize_coordinates(polygons, width, height)
                
                for polygon in normalized_polygons:
                    polygon_str = ' '.join(map(str, polygon))
                    f.write(f"{class_id} {polygon_str}\n")

# Example usage
base_path = 'dl_challenge/'
create_yolo_files(base_path)
