import argparse
import numpy as np
import os
from tqdm import tqdm
import PIL
from PIL import Image

import mmcv
import mmengine.fileio as fileio


def dataset_cleaning(args):

    images = []
    with open(os.path.join(args.input_dir, args.train_val_dataset), "r") as f:
        for line in f:
            images.append(os.path.join(args.input_dir, args.ori_jpg_name, line.strip() + '.jpg'))
    
    for idx in tqdm(range(len(images))):
        
        # filename = "/yuanhuan/data/image/RM_Capture/original_dupes_0_95/JPEGImages/2133-3905176-0_jiAY2100.jpg"
        filename = images[idx]
        img_bytes = fileio.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
        img = img.astype(np.float32)
        print(filename)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Capture
    args.input_dir = "/yuanhuan/data/image/RM_Capture/original_dupes_0_95"
    args.ori_jpg_name = "JPEGImages"
    args.train_val_dataset = "ImageSets/Main/test.txt"

    dataset_cleaning(args)