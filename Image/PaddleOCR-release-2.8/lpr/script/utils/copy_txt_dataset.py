import argparse
import os
import shutil
from tqdm import tqdm

def copy_dataset(args):
    # mkdir 
    if not os.path.exists(args.jpg_output_dir):
        os.makedirs(args.jpg_output_dir)

    with open(args.test_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            jpg_path = line.strip()
            jpg_name = os.path.basename(jpg_path)
            output_jpg_path = os.path.join(args.jpg_output_dir, jpg_name)
            shutil.copy(jpg_path, output_jpg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir =  "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_all_UAE/"

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"

    args.jpg_dir = os.path.join(args.input_dir, "Images")
    args.jpg_output_dir = os.path.join(args.input_dir, "Images_test")

    copy_dataset(args)
