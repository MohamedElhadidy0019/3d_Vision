import cv2
import os
import numpy as np

# h max is 180 while s,v max is 255
# z value is between 0.8 and 1.3

def rgb_to_hvd(rgb_path, pc_path, folder_path):
    img = cv2.imread(rgb_path)
    # to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    pc = np.load(pc_path)
    z = pc[2]
    z = np.clip(z, 0.8, 1.3)

    h_norm = h / 180.0
    v_norm = v / 255.0
    z_norm = (z - 0.8) / (1.3 - 0.8)

    # concat all
    hvz = np.stack([h_norm, v_norm, z_norm], axis=0)
    hvz = np.transpose(hvz, (1, 2, 0))
    #    cv2.imwrite(os.path.join(destination_path, 'rgbd.png'), rgbd)
    # get the path of rbg exepect the image name
    cv2.imwrite(os.path.join(folder_path, 'hvz.png'), hvz)


    return hvz




def main():
    ds_path = 'dl_challenge/'

    for folder in os.listdir(ds_path):
        rgb_path = os.path.join(ds_path, folder, 'rgb.jpg')
        pc_path = os.path.join(ds_path, folder, 'pc.npy')
        # print(rgb_path, pc_path)

        hvz_image =rgb_to_hvd(rgb_path, pc_path, os.path.join(ds_path, folder))  
        # save it as png
        




if __name__ == '__main__':
    main()