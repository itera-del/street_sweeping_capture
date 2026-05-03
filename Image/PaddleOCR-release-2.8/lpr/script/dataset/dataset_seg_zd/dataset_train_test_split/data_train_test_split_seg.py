import argparse
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    # mkdir
    create_folder(os.path.dirname(args.trainval_file))

    jpg_list = np.array(os.listdir(args.img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.mask_dir, jpg.replace(".jpg", args.mask_suffix))) for jpg in jpg_list]]

    trainval_list, test_list = train_test_split(jpg_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)

    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))

    with open(args.trainval_file, "w") as f:
        for jpg in trainval_list:
            f.write('{} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.test_file, "w") as f:
        for jpg in test_list:
            f.write('{} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.train_file, "w") as f:
        for jpg in train_list:
            f.write('{} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.val_file, "w") as f:
        for jpg in val_list:
            f.write('{} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")


def split_2_softmax(args):
    # mkdir
    create_folder(os.path.dirname(args.trainval_file))

    jpg_list = np.array(os.listdir(args.img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.one_mask_dir, jpg.replace(".jpg", args.mask_suffix))) for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.two_mask_dir, jpg.replace(".jpg", args.mask_suffix))) for jpg in jpg_list]]

    trainval_list, test_list = train_test_split(jpg_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)

    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))

    with open(args.trainval_file, "w") as f:
        for jpg in trainval_list:
            f.write('{} {} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.one_mask_dir, jpg.replace(".jpg", args.mask_suffix)), os.path.join(args.two_mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.test_file, "w") as f:
        for jpg in test_list:
            f.write('{} {} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.one_mask_dir, jpg.replace(".jpg", args.mask_suffix)), os.path.join(args.two_mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.train_file, "w") as f:
        for jpg in train_list:
            f.write('{} {} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.one_mask_dir, jpg.replace(".jpg", args.mask_suffix)), os.path.join(args.two_mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")

    with open(args.val_file, "w") as f:
        for jpg in val_list:
            f.write('{} {} {}'.format(os.path.join(args.img_dir, jpg), os.path.join(args.one_mask_dir, jpg.replace(".jpg", args.mask_suffix)), os.path.join(args.two_mask_dir, jpg.replace(".jpg", args.mask_suffix))))
            f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/") 
    args = parser.parse_args()

    print("5、data train test split seg.")
    print("input_dir: {}".format(args.input_dir))

    ###############################################
    # dataset_zd_dict
    ###############################################

    args.img_dir = os.path.join(args.input_dir, "Images")
    args.mask_dir = os.path.join(args.input_dir, "city_label/mask")
    args.mask_suffix = ".png"
    
    args.trainval_file = os.path.join(args.input_dir, "city_label/ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, "city_label/ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, "city_label/ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, "city_label/ImageSets/Main/test.txt")

    args.test_size = 0.1
    args.val_size = 0.1
    
    split(args)

    ###############################################
    # dataset_zd_dict_color
    ###############################################

    args.img_dir = os.path.join(args.input_dir, "Images")
    args.mask_dir = os.path.join(args.input_dir, "color_label/mask")
    args.mask_suffix = ".png"
    
    args.trainval_file = os.path.join(args.input_dir, "color_label/ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, "color_label/ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, "color_label/ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, "color_label/ImageSets/Main/test.txt")

    args.test_size = 0.1
    args.val_size = 0.1
    
    split(args)

    ###############################################
    # dataset_zd_dict & dataset_zd_dict_color
    ###############################################

    args.img_dir = os.path.join(args.input_dir, "Images")
    args.one_mask_dir = os.path.join(args.input_dir, "city_label/mask")
    args.two_mask_dir = os.path.join(args.input_dir, "color_label/mask")
    args.mask_suffix = ".png"
    
    args.trainval_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/test.txt")

    args.test_size = 0.1
    args.val_size = 0.1
    
    split_2_softmax(args)