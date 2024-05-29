import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt



def rgb_to_rgbd(rgb_path, pc_path, destination_path):
    rgb = Image.open(rgb_path)
    # plt.imshow(rgb)
    # plt.show()
    # plt.close()
    rgb_np = np.array(rgb)
    # plt.imshow(rgb_np)
    # plt.show()
    # plt.close()
    pc = np.load(pc_path)
    pc_z = pc[2]
    pc_z = np.clip(pc_z, 0.8, 1.3)
    pc_z = (pc_z - 0.8) / 0.5
    pc_z = pc_z * 255
    pc_z = np.expand_dims(pc_z, axis=2)
    # type cast pc_z to uint8
    pc_z = pc_z.astype(np.uint8)
    
    # concat rgb_np and pc_z
    rgbd = np.concatenate((rgb_np, pc_z), axis=2)
    # depth_map = rgbd[:, :, 3]
    # img_np = rgbd[:, :, :3]
    # plt.imshow(img_np)
    # plt.show()
    # plt.close()
    # plt.imshow(depth_map, cmap='gray')
    # plt.show()
    # plt.close()

    # save as png rgbd.png
    
    # 

    cv2.imwrite(os.path.join(destination_path, 'rgbd.png'), rgbd)

    # rgbd= cv2.imread(os.path.join(destination_path, 'rgbd.png'), cv2.IMREAD_UNCHANGED)
    # # show the image
    # rgb = rgbd[:, :, :3]
    # depth = rgbd[:, :, 3]
    # # cv2.imshow('image', rgb)
    # plt.imshow(rgb)
    # plt.show()
    # plt.close()

    pass







def main():
    path = 'dl_challenge/'
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        pc_path = os.path.join(folder_path, 'pc.npy')
        rgb_to_rgbd(rgb_path, pc_path, folder_path)








if __name__ == '__main__':
    main()