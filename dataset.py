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
from torchvision.transforms import v2 as T

def is_valid_box(box):
    x_min, y_min, x_max, y_max = box
    return (x_max > x_min) and (y_max > y_min)

def filter_invalid_boxes(target):
    # for target in targets:
    valid_boxes = []
    valid_labels = []
    for box, label in zip(target['boxes'], target['labels']):
        if is_valid_box(box):
            valid_boxes.append(box)
            valid_labels.append(label)
    target['boxes'] = torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4))
    target['labels'] = torch.tensor(valid_labels) if valid_labels else torch.empty((0,), dtype=torch.int64)
    return target

class ChallengeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, print_name=False):
        self.root = root
        self.transforms = transforms
        self.print_name = print_name
        self.data_paths = list(sorted(os.listdir(root)))

    def __getitem__(self, index):
        current_path = self.data_paths[index]
        if self.print_name:
            print(f'current_path: {current_path}')  
        # print(f'current_path: {current_path}')
        img_path = os.path.join(self.root, current_path, 'rgb.jpg')
        mask_path = os.path.join(self.root, current_path, 'mask.npy')
        pc_path = os.path.join(self.root, current_path, 'pc.npy')

        img = Image.open(img_path)
        img_np = np.array(img)
        img_np = img_np.transpose((2, 0, 1))  # now it has wanted shape
        
        pc_npy = np.load(pc_path)
        pc_npy[2,:,:] = np.clip(pc_npy[2,:,:], 0.8, 1.3)
        pc_npy = pc_npy.astype(np.float32)
        pc_z = pc_npy[2,:,:]
        # add dim 1 to the pc_z
        pc_z = np.expand_dims(pc_z, axis=0)

        # img from uint8 to float64
        img_np = img_np.astype(np.float32)
        # concatenate img and pc
        img_w_pc = np.concatenate((img_np, pc_z), axis=0)
        img_w_pc = torch.from_numpy(img_w_pc).to(torch.float32)

        # to tensor
        masks = np.load(mask_path)
        bacgkround = np.all(masks == False, axis=0)
        background = np.expand_dims(bacgkround, axis=0)
        masks = np.concatenate([background, masks], axis=0)
        mask = np.argmax(masks, axis=0)
        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask).to(torch.uint8)

         # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = index
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img_w_pc = tv_tensors.Image(img_w_pc)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img_w_pc, target = self.transforms(img_w_pc, target)
        boxes_list = []
        for box in target['boxes']:
            if is_valid_box(box):
                boxes_list.append(box)
        target['boxes'] = torch.stack(boxes_list)
        return img_w_pc, target


    def __len__(self):
        return len(self.data_paths)
    
# def visualize_image_with_boxes(img, boxes,masks, index = 0):
#     """
#     Visualizes an image along with its bounding boxes.
    
#     Args:
#         img (torch.Tensor or tv_tensors.Image): The image tensor with 6 channels (RGB + point cloud).
#         boxes (torch.Tensor or tv_tensors.BoundingBoxes): The bounding boxes tensor in XYXY format.
#     """
#     # Check if img is a tv_tensors.Image and convert it to a tensor if necessary
#     if isinstance(img, tv_tensors.Image):
#         # img = img.as_subclass(torch.Tensor)
#         img = torch.Tensor(img)
#         masks = torch.Tensor(masks)

#     masks_np = masks.numpy()
#     masks_np = np.argmax(masks_np, axis=0)

#     # Convert the image tensor to numpy for visualization
#     img_np = img[:3].numpy().transpose((1, 2, 0))  # Only take the first 3 channels (RGB) and reshape

#     # Ensure the image is in the range [0, 255] for proper display
#     img_np = img_np.astype(np.uint8)

#     # Print debug information
#     # print(f"Image shape: {img_np.shape}")
#     # print(f"Boxes: {boxes}")

#     # Create a figure and axis
#     fig, ax = plt.subplots(1, figsize=(12, 12))
    
#     # Display the image
#     ax.imshow(img_np)
    
#     # Plot each bounding box
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#     # save the plot
#     plt.show()
#     # plt.savefig(f'bbox_pics/{index:03d}.png')
#     plt.close(fig)
#     print(f'bbox_pics/{index:03d}.png saved')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_boxes(img, boxes, masks, index=0):
    """
    Visualizes an image along with its bounding boxes and the corresponding mask.
    
    Args:
        img (torch.Tensor or tv_tensors.Image): The image tensor with 6 channels (RGB + point cloud).
        boxes (torch.Tensor or tv_tensors.BoundingBoxes): The bounding boxes tensor in XYXY format.
        masks (torch.Tensor): The mask tensor.
        index (int): The index used for saving the image file.
    """
    # Check if img is a tv_tensors.Image and convert it to a tensor if necessary
    if isinstance(img, tv_tensors.Image):
        img = torch.Tensor(img)
        masks = torch.Tensor(masks)

    masks_np = masks.numpy()
    background = np.all(masks_np == False, axis=0)
    background = np.expand_dims(background, axis=0)
    masks_np = np.concatenate([background, masks_np], axis=0)
    masks_np = np.argmax(masks_np, axis=0)

    # Convert the image tensor to numpy for visualization
    img_np = img[:3].numpy().transpose((1, 2, 0))  # Only take the first 3 channels (RGB) and reshape

    # Ensure the image is in the range [0, 255] for proper display
    img_np = img_np.astype(np.uint8)

    # Create a figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))  # Create 1 row, 2 columns of subplots
    
    # Display the image
    axes[0].imshow(img_np)
    axes[0].set_title("Image with Bounding Boxes")
    
    # Plot each bounding box on the first subplot
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)

    # Display the mask on the second subplot
    cmap_mask = plt.cm.get_cmap('tab20')

    axes[1].imshow(masks_np, cmap=cmap_mask)
    axes[1].set_title("Mask")
    
    # Save the plot
    plt.show()
    # plt.savefig(f'bbox_pics/{index:03d}.png')
    plt.close(fig)
    print(f'bbox_pics/{index:03d}.png saved')

# Example usage
# img = torch.rand(6, 256, 256)  # Example image tensor with 6 channels
# boxes = torch.tensor([[30, 30, 200, 200], [50, 50, 100, 100]])  # Example bounding boxes
# masks = torch.rand(1, 256, 256)  # Example masks tensor
# visualize_image_with_boxes(img, boxes, masks)

def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    # resize 256,256
    transforms.append(T.Resize((384, 512)))
    transforms.append(T.ToDtype(torch.float))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def main():
    dataset = ChallengeDataset(root='dl_challenge/', transforms=get_transform())
    print(len(dataset))
    for i in range(5, len(dataset)):
        img, target = dataset[i]
        boxes = target['boxes']
        masks = target['masks']

        visualize_image_with_boxes(img, boxes,masks, i)


        # break
if __name__ == '__main__':
    main()