import numpy as np
from PIL import Image


# ratio_list = []
# for i in range(200):
#     image_path = f'dl_challenge/{i:03d}/rgb.jpg'
#     image = Image.open(image_path)
#     image_np = np.array(image)
#     ratio = image_np.shape[0] / image_np.shape[1]
#     print(f'image_np.shape = {image_np.shape}')
#     print(f'ratio = {ratio}')
#     ratio_list.append(ratio)

z_values = []
for i in range(200):
    pc_path = f'dl_challenge/{i:03d}/pc.npy'
    pc_npy = np.load(pc_path)
    pc_z = pc_npy[2,:,:]
    # clip
    pc_npy[2,:,:] = np.clip(pc_npy[2,:,:], 0.8, 1.3)
    pc_z = pc_npy[2,:,:]
    # flatten
    pc_z = pc_z.flatten()
    # to list
    pc_z = pc_z.tolist()
    # add each single value to the list
    for z in pc_z:
        z_values.append(z)

    # z_values.append(pc_z)


# # get avg without statistiacl outliers
# ratio_list = np.array(ratio_list)
# avg_ratio = np.mean(ratio_list)
# print(f'avg_ratio = {avg_ratio}')
# get mean and standard deviation of z values
z_values = np.array(z_values)
z_values_mean = np.mean(z_values)
z_values_std = np.std(z_values)
print(f'z_values_mean = {z_values_mean}')
print(f'z_values_std = {z_values_std}')
