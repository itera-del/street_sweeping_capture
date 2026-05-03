import argparse
import shutil
import os
import sys
from tqdm import tqdm


# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def dataset_split(args):
    
    # load file
    file_list = []
    with open(args.all_file, "r") as f:
        for line in f:
            file_list.append(str(line).strip())
    print("File Len: ", len(file_list))

    # load ignore list
    ignore_file_list = []
    for idx in range(len(args.ignore_dir_list)):
        ignore_dir_idx = os.path.join(args.output_dir, args.ignore_dir_list[idx])
        ignore_file_list.extend(os.listdir(ignore_dir_idx))
    print("Ignore File Len: ", len(ignore_file_list))

    # find num
    find_num = 0

    for idx in tqdm(range(len(file_list))):
        
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)

        if file_name not in ignore_file_list:

            find_num += 1
            tqdm.write("find_num: {}".format(find_num))

            output_path = os.path.join(args.output_dir, args.output_format.format(args.split_id), file_name)

            # mkdir
            create_folder(os.path.dirname(output_path))

            shutil.copy(file_path, output_path)

        if find_num >= args.split_nums:
            break

            
def dataset_split_color(args):
    
    # load file
    file_list = []
    with open(args.all_file, "r") as f:
        for line in f:
            file_list.append(str(line).strip())
    print("File Len: ", len(file_list))

    # load ignore list
    ignore_file_list = []
    for idx in range(len(args.ignore_dir_list)):
        ignore_dir_idx = os.path.join(args.output_dir, args.ignore_dir_list[idx])
        ignore_file_list.extend(os.listdir(ignore_dir_idx))
    print("Ignore File Len: ", len(ignore_file_list))

    # find num
    find_num = 0

    for idx in tqdm(range(len(file_list))):
        
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)
        file_label_list = str(file_name).split('.jpg')[0].split('-')[-1].split('_')
        file_label_color = file_label_list[3]

        if file_name not in ignore_file_list and \
            file_label_color == args.split_color:

            find_num += 1
            tqdm.write("find_num: {}".format(find_num))

            output_path = os.path.join(args.output_dir, args.output_format.format(args.split_color, args.split_id), file_name)

            # mkdir
            create_folder(os.path.dirname(output_path))

            shutil.copy(file_path, output_path)

        if find_num >= args.split_nums:
            break


def dataset_split_city(args):
    
    # load file
    file_list = []
    with open(args.all_file, "r") as f:
        for line in f:
            file_list.append(str(line).strip())
    print("File Len: ", len(file_list))

    # load ignore list
    ignore_file_list = []
    for idx in range(len(args.ignore_dir_list)):
        ignore_dir_idx = os.path.join(args.output_dir, args.ignore_dir_list[idx])
        ignore_file_list.extend(os.listdir(ignore_dir_idx))
    print("Ignore File Len: ", len(ignore_file_list))

    # find num
    find_num = 0

    for idx in tqdm(range(len(file_list))):
        
        file_path = file_list[idx]
        file_name = os.path.basename(file_path)
        file_label_list = str(file_name).split('.jpg')[0].split('-')[-1].split('_')
        file_label_city = file_label_list[2]

        if file_name not in ignore_file_list and \
            file_label_city == args.split_city:

            find_num += 1
            tqdm.write("find_num: {}".format(find_num))

            output_path = os.path.join(args.output_dir, args.output_format.format(args.split_city, args.split_id), file_name)

            # mkdir
            create_folder(os.path.dirname(output_path))

            shutil.copy(file_path, output_path)

        if find_num >= args.split_nums:
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.ignore_dir_list = [
        "check_crop_0/Images/",
        "check_crop_1/Images/",
        "check_crop_2/Images/",
        "check_crop_3/Images/",
        "check_crop_4/Images/",
        "check_crop_5/Images/",
        "check_crop_6/Images/",
        "check_crop_7/Images/",
        "check_crop_8/Images/",
        "check_crop_color_green_0/Images/",
        "check_crop_color_yellow_0/Images/",
        "check_crop_city_RAK_0/Images/",
        "check_crop_city_SHARJAH_0/Images/",
        "check_crop_city_UMMALQAIWAIN_0/Images/",
        "check_crop_city_FUJAIRAH_0/Images/",
        "check_crop_city_AJMAN_0/Images/",
    ]

    ###############################################
    ## normal
    ###############################################
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop/"
    args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/"
    args.output_format = "check_crop_{}/Images/"

    args.all_file = args.input_dir + "ImageSets/Main/all.txt"
    args.split_nums = 20000
    args.split_id = 8

    dataset_split(args)

    # ###############################################
    # ## color
    # ###############################################
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop/"
    # args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/"
    # args.output_format = "check_crop_color_{}_{}/Images/"

    # args.all_file = args.input_dir + "ImageSets/Main/all.txt"
    # args.split_nums = 10000
    # # args.split_color = "green"
    # # args.split_id = 0
    # args.split_color = "yellow"
    # args.split_id = 0

    # dataset_split_color(args)

    # ###############################################
    # ## city
    # ###############################################
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop/"
    # args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/"
    # args.output_format = "check_crop_city_{}_{}/Images/"

    # args.all_file = args.input_dir + "ImageSets/Main/all.txt"
    # args.split_nums = 10000
    # # args.split_city = "RAK"
    # # args.split_id = 0
    # # args.split_city = "SHARJAH"
    # # args.split_id = 0
    # # args.split_city = "UMMALQAIWAIN"
    # # args.split_id = 0
    # # args.split_city = "FUJAIRAH"
    # # args.split_id = 0
    # args.split_city = "AJMAN"
    # args.split_id = 0

    # dataset_split_city(args)