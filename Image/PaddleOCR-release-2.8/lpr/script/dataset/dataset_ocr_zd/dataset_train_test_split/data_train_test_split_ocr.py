import argparse
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    # mkdir
    create_folder(os.path.dirname(args.trainval_file))

    jpg_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    jpg_list.sort()

    trainval_list, test_list = train_test_split(jpg_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)

    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))

    with open(args.all_file, "w") as f:
        for jpg in jpg_list:
            f.write('{}'.format(jpg))
            f.write("\n")

    with open(args.trainval_file, "w") as f:
        for jpg in trainval_list:
            f.write('{}'.format(jpg))
            f.write("\n")

    with open(args.test_file, "w") as f:
        for jpg in test_list:
            f.write('{}'.format(jpg))
            f.write("\n")

    with open(args.train_file, "w") as f:
        for jpg in train_list:
            f.write('{}'.format(jpg))
            f.write("\n")

    with open(args.val_file, "w") as f:
        for jpg in val_list:
            f.write('{}'.format(jpg))
            f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/data_crop/") 
    args = parser.parse_args()

    print("add、data train test split ocr.")
    print("input_dir: {}".format(args.input_dir))

    args.img_dir = os.path.join(args.input_dir, "Images")
    args.all_file = os.path.join(args.input_dir, "ImageSets/Main/all.txt")
    args.trainval_file = os.path.join(args.input_dir, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, "ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, "ImageSets/Main/test.txt")

    args.test_size = 0.05
    args.val_size = 0.05

    split(args)