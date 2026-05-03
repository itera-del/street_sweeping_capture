import argparse
import os
import random
import shutil
from tqdm import tqdm

def copy_dataset(args):
    # mkdir 
    if not os.path.exists(args.jpg_output_dir):
        os.makedirs(args.jpg_output_dir)

    # with open(args.trainval_file, "r") as f:
    with open(args.train_file, "r") as f:
    # with open(args.val_file, "r") as f:
    # with open(args.test_file, "r") as f:
    # with open(args.split_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            
            jpg_path = line.strip().split('.jpg')[0] + '.jpg'
            jpg_name = os.path.basename(jpg_path)
            output_jpg_path = os.path.join(args.jpg_output_dir, jpg_name)
            
            if random.random() >= args.random_scale:
                print(jpg_path)
                print(output_jpg_path)
                shutil.copy(jpg_path, output_jpg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/plate_zd_mask_202502") 
    parser.add_argument('--split_dir', type=str, default="ImageSetsOcrLabelNoAugNoDiffste") 
    args = parser.parse_args()

    args.trainval_file = os.path.join(args.input_dir, args.split_dir, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, args.split_dir, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, args.split_dir, "ImageSets/Main/val.txt") 
    args.test_file = os.path.join(args.input_dir, args.split_dir, "ImageSets/Main/test.txt")
    args.split_file = os.path.join(args.input_dir, args.split_dir, "ImageSets/Main/exclude_split_0_1.txt")

    args.jpg_output_dir = os.path.join(args.input_dir, "quantization_data")

    # args.random_scale = 0.0
    args.random_scale = (1 - 0.001)

    copy_dataset(args)
