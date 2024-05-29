

import os



def yolo_to_ds(origin_path, destination_path):
    

    for folder in os.listdir(origin_path):
        folder_path = os.path.join(origin_path, folder)
        image_path = os.path.join(folder_path, 'rgbd.png')
        txt_file_path = os.path.join(folder_path, 'mask.txt')

        # copy image to destination, name the image folder_path.jpg and save it in destination_path/images
        os.system(f'cp {image_path} {destination_path}/images/{folder}.png')
        # copy txt_file to destination, name the txt_file folder_path.txt and save it in destination_path/labels
        os.system(f'cp {txt_file_path} {destination_path}/labels/{folder}.txt')
        








def main():
    path = 'dl_challenge/'
    destination_path = 'yolo_ds_rgbd/train'
    yolo_to_ds(path, destination_path)



if __name__ == '__main__':
    main()

