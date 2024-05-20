import os
import torch

import torch.utils
import torch.utils.data
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ChallengeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.data_paths = list(sorted(os.listdir(root)))

    def __getitem__(self, index):
        current_path = self.data_paths[index]
        print(f'current_path: {current_path}')
        img_path = os.path.join(self.root, current_path, 'rgb.jpg')
        mask_path = os.path.join(self.root, current_path, 'mask.npy')
        pc_path = os.path.join(self.root, current_path, 'pc.npy')

        img = Image.open(img_path)
        img_np = np.array(img)
        img_np = img_np.transpose((2, 0, 1))  # now it has wanted shape
        pc_npy = np.load(pc_path)
        pc_npy[2,:,:] = np.clip(pc_npy[2,:,:], 0.8, 1.3)

        # img from uint8 to float64
        img_np = img_np.astype(np.float64)
        # concatenate img and pc
        img_w_pc = np.concatenate((img_np, pc_npy), axis=0)
        img_w_pc = torch.from_numpy(img_w_pc).to(torch.float64)

        # to tensor
        masks = np.load(mask_path)
        num_objs = masks.shape[0]
        masks = torch.from_numpy(masks).to(torch.uint8)

        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = index
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img_w_pc)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.data_paths)
    
def visualize_image_with_boxes(img, boxes):
    """
    Visualizes an image along with its bounding boxes.
    
    Args:
        img (torch.Tensor or tv_tensors.Image): The image tensor with 6 channels (RGB + point cloud).
        boxes (torch.Tensor or tv_tensors.BoundingBoxes): The bounding boxes tensor in XYXY format.
    """
    # Check if img is a tv_tensors.Image and convert it to a tensor if necessary
    if isinstance(img, tv_tensors.Image):
        # img = img.as_subclass(torch.Tensor)
        img = torch.Tensor(img)


    # Convert the image tensor to numpy for visualization
    img_np = img[:3].numpy().transpose((1, 2, 0))  # Only take the first 3 channels (RGB) and reshape

    # Ensure the image is in the range [0, 255] for proper display
    img_np = img_np.astype(np.uint8)

    # Print debug information
    print(f"Image shape: {img_np.shape}")
    print(f"Boxes: {boxes}")

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    # Display the image
    ax.imshow(img_np)
    
    # Plot each bounding box
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()


# def main():
#     dataset = ChallengeDataset(root='dl_challenge/')
#     print(len(dataset))
#     for i in range(100):
#         img, target = dataset[i]
#         boxes = target['boxes']
#         visualize_image_with_boxes(img, boxes)


#         # break
# if __name__ == '__main__':
#     main()