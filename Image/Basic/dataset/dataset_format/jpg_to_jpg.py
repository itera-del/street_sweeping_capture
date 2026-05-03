import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
from tqdm import tqdm
import shutil


def bmp_to_jpg(args):
    img_list = np.array(os.listdir(args.jpg_dir))
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.jpg_dir, img_list[idx])

        if img_path.endswith('.bmp'):
            img = cv2.imread(img_path, -1)
            cv2.imwrite(img_path.replace('.bmp','.jpg'), img)
            os.remove(img_path)
        elif img_path.endswith('.jpeg'):
            img = cv2.imread(img_path, -1)
            cv2.imwrite(img_path.replace('.jpeg','.jpg'), img)
            os.remove(img_path)
        elif img_path.endswith('.png'):
            img = cv2.imread(img_path, -1)
            cv2.imwrite(img_path.replace('.png','.jpg'), img)
            os.remove(img_path)
        elif img_path.endswith('.JPG'):
            img = cv2.imread(img_path, -1)
            cv2.imwrite(img_path.replace('.JPG','.jpg'), img)
            os.remove(img_path)
        elif img_path.endswith('.jpg'):
            img = cv2.imread(img_path, -1)
            output_img_path = os.path.join(args.output_jpg_dir, img_list[idx])
            cv2.imwrite(output_img_path, img)
        elif not img_path.endswith('.jpg'):
            print(img_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpg_dir", type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear")
    parser.add_argument("--output_jpg_dir", type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear")
    args = parser.parse_args()

    if not os.path.exists(args.output_jpg_dir):
        os.makedirs(args.output_jpg_dir)
        
    bmp_to_jpg(args)
