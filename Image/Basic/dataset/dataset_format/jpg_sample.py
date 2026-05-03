import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, "/yuanhuan/code/demo")
from Image.Basic.utils.folder_tools import *


def jpg_sample(args):
    # mkdir
    create_folder(args.output_dir)

    # get folder name as prefix
    folder_name = os.path.basename(os.path.normpath(args.input_dir))

    # get all jpg images
    image_list = np.array(os.listdir(args.input_dir))
    image_list = image_list[[img.endswith('.jpg') or img.endswith('.JPG') for img in image_list]]
    
    # sort images by numeric order (extract number from filename)
    def get_image_number(filename):
        try:
            # Extract number from filename (e.g., "0.jpg" -> 0, "100.jpg" -> 100)
            return int(os.path.splitext(filename)[0])
        except ValueError:
            return 0
    
    image_list = sorted(image_list, key=get_image_number)
    image_list = np.array(image_list)
    
    print(f"Total images: {len(image_list)}")
    print(f"Frame step: {args.frame_strp}")
    print(f"Output images: {len(image_list) // args.frame_strp + (1 if len(image_list) % args.frame_strp > 0 else 0)}")

    # sample and copy images
    for idx in tqdm(range(len(image_list))):
        if idx % args.frame_strp == 0:
            input_img_path = os.path.join(args.input_dir, image_list[idx])
            # add folder name as prefix to output filename
            original_filename = image_list[idx]
            name, ext = os.path.splitext(original_filename)
            output_filename = f"{folder_name}_{name}{ext}"
            output_img_path = os.path.join(args.output_dir, output_filename)
            
            # read and write image
            img = cv2.imread(input_img_path)
            if img is not None:
                cv2.imwrite(output_img_path, img)
            else:
                print(f"Warning: Failed to read image {input_img_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/AEB/Calibrate/Chessboard/B11E02_20260123/h264_analyze/h264_data/CH1-260123-102600-102730",
        # default="/yuanhuan/data/image/AEB/Calibrate/Chessboard/B11E02_20260123/h264_analyze/h264_data/CH1-260123-103100-103300",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/yuanhuan/data/image/AEB/Calibrate/Chessboard/B11E02_20260123/calibrate_img",
    )
    parser.add_argument(
        "--frame_strp",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    jpg_sample(args)
