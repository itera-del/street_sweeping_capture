import argparse
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def get_copy_list(file_path):

    file_list = []
    with open(file_path, "r") as f:
        for line in f:

            filename = os.path.basename(str(line).strip().split('.jpg')[0])

            filename_split_list = filename.split("_")[:-1]
            file_list.append(("-").join(filename_split_list) + '.jpg')
    
    return file_list


def get_list(jpg_list, copy_train_list, copy_val_list, copy_test_list, to_img_dir):

    trainval_list, train_list, val_list, test_list = [], [], [], []

    for idx in tqdm(range(len(jpg_list))):
        
        jpg_path = jpg_list[idx]
        jpg_name = os.path.basename(jpg_path)
        
        jpg_name_split_list = jpg_name.strip().split('.jpg')[0].split("_")[:-1]
        jpg_id = ("-").join(jpg_name_split_list) + '.jpg'

        add_line = '{}'.format(os.path.join(to_img_dir, jpg_name))

        if jpg_id in copy_train_list and \
            jpg_id in copy_val_list:
            # raise Exception
            continue

        if jpg_id in copy_train_list and \
            jpg_id in copy_test_list:
            # raise Exception
            continue

        if jpg_id in copy_val_list and \
            jpg_id in copy_test_list:
            # raise Exception
            continue

        if jpg_id in copy_train_list:
            train_list.append(add_line)
            trainval_list.append(add_line)

        elif jpg_id in copy_val_list:
            val_list.append(add_line)
            trainval_list.append(add_line)

        elif jpg_id in copy_test_list:
            test_list.append(add_line)
        
        else:
            # raise Exception
            continue

    return trainval_list, train_list, val_list, test_list


def split(args):
    
    # mkdir
    create_folder(os.path.dirname(args.to_trainval_file))

    # jpg_list
    jpg_list = get_sub_filepaths_suffix(args.to_img_dir, ".jpg")
    jpg_list.sort()

    # copy_list
    copy_train_list = get_copy_list(args.copy_from_train_file)
    copy_val_list = get_copy_list(args.copy_from_val_file)
    copy_test_list = get_copy_list(args.copy_from_test_file)

    # find list
    trainval_list, train_list, val_list, test_list = get_list( jpg_list, \
                                                                copy_train_list, copy_val_list, copy_test_list, \
                                                                args.to_img_dir )

    with open(args.to_trainval_file, "w") as f:
        for line in trainval_list:
            f.write('{}'.format(line))
            f.write("\n")

    with open(args.to_test_file, "w") as f:
        for line in test_list:
            f.write('{}'.format(line))
            f.write("\n")

    with open(args.to_train_file, "w") as f:
        for line in train_list:
            f.write('{}'.format(line))
            f.write("\n")

    with open(args.to_val_file, "w") as f:
        for line in val_list:
            f.write('{}'.format(line))
            f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_1115_1116/") 
    args = parser.parse_args()

    print("5、data train test split ocr copy.")
    print("input_dir: {}".format(args.input_dir))

    ###############################################
    # normal
    ###############################################
    args.copy_from_dir = args.input_dir

    args.copy_from_trainval_file = os.path.join(args.copy_from_dir, "city_label/ImageSets/Main/trainval.txt")
    args.copy_from_train_file = os.path.join(args.copy_from_dir, "city_label/ImageSets/Main/train.txt")
    args.copy_from_val_file = os.path.join(args.copy_from_dir, "city_label/ImageSets/Main/val.txt")
    args.copy_from_test_file = os.path.join(args.copy_from_dir, "city_label/ImageSets/Main/test.txt")

    args.to_dir = args.output_dir

    args.to_img_dir = os.path.join(args.to_dir, "Images")
    args.to_trainval_file = os.path.join(args.to_dir, "ImageSets/Main/trainval.txt")
    args.to_train_file = os.path.join(args.to_dir, "ImageSets/Main/train.txt")
    args.to_val_file = os.path.join(args.to_dir, "ImageSets/Main/val.txt")
    args.to_test_file = os.path.join(args.to_dir, "ImageSets/Main/test.txt")

    split(args)

    ###############################################
    # augment
    ###############################################

    args.copy_from_dir = args.input_dir

    args.copy_from_trainval_file = os.path.join(args.copy_from_dir, "city_label/ImageSets_aug/ImageSets/Main/trainval.txt")
    args.copy_from_train_file = os.path.join(args.copy_from_dir, "city_label/ImageSets_aug/ImageSets/Main/train.txt")
    args.copy_from_val_file = os.path.join(args.copy_from_dir, "city_label/ImageSets_aug/ImageSets/Main/val.txt")
    args.copy_from_test_file = os.path.join(args.copy_from_dir, "city_label/ImageSets_aug/ImageSets/Main/test.txt")

    args.to_dir = args.output_dir

    args.to_img_dir = os.path.join(args.to_dir, "Images_aug")
    args.to_trainval_file = os.path.join(args.to_dir, "ImageSets_aug/ImageSets/Main/trainval.txt")
    args.to_train_file = os.path.join(args.to_dir, "ImageSets_aug/ImageSets/Main/train.txt")
    args.to_val_file = os.path.join(args.to_dir, "ImageSets_aug/ImageSets/Main/val.txt")
    args.to_test_file = os.path.join(args.to_dir, "ImageSets_aug/ImageSets/Main/test.txt")

    split(args)