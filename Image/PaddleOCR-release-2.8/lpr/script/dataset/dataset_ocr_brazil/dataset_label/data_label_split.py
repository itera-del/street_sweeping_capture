import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def data_label_split(args):

    # split
    for idx in range(len(args.split_file_list)):

        split_file_idy = args.split_file_list[idx]
        input_split_file_idy = os.path.join(args.split_dir, split_file_idy)
        output_first_split_file_idy = os.path.join(args.output_first_split_dir, split_file_idy)
        output_second_split_file_idy = os.path.join(args.output_second_split_dir, split_file_idy)

        # mkdir
        create_folder(os.path.dirname(output_first_split_file_idy))
        create_folder(os.path.dirname(output_second_split_file_idy))

        file_list = []
        with open(input_split_file_idy, "r") as f:
            for line in f:
                file_list.append(line.strip())

        with open(output_first_split_file_idy, "w") as f:
            for file_path in file_list:
                
                file_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                file_num = str(file_name).replace('.jpg', '').split('_')[-1]
                file_first_num = str(file_name).replace('.jpg', '').split('_')[-1][:3]

                out_file_path = os.path.join(file_dir, str(file_name).replace(file_num, file_first_num))
                # print("{} -> {}".format(file_path, out_file_path))
                f.write('{}'.format(out_file_path))
                f.write("\n")

        with open(output_second_split_file_idy, "w") as f:
            for file_path in file_list:
                
                file_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                file_num = str(file_name).replace('.jpg', '').split('_')[-1]
                file_scrond_num = str(file_name).replace('.jpg', '').split('_')[-1][3:]

                out_file_path = os.path.join(file_dir, str(file_name).replace(file_num, file_scrond_num))
                # print("{} -> {}".format(file_path, out_file_path))
                f.write('{}'.format(out_file_path))
                f.write("\n")


if __name__ == "__main__":

    ###
    # 针对摩托车车牌，拆分第一行、第二行数据
    ###

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_brazil/") 
    args = parser.parse_args()

    args.split_file_list = [
                            "ImageSets/Main/trainval.txt", 
                            "ImageSets/Main/train.txt",
                            "ImageSets/Main/val.txt",
                            "ImageSets/Main/test.txt",
                            ]

    print("5、data label split.")
    print("input_dir: {}".format(args.input_dir))


    ########################
    # no aug
    ########################

    args.split_dir = os.path.join(args.input_dir, "ImageSetsNoAug")
    args.output_first_split_dir = os.path.join(args.split_dir + "_1st_line")
    args.output_second_split_dir = os.path.join(args.split_dir + "_2nd_line")

    data_label_split(args)

    ########################
    # with aug 
    ########################

    args.split_dir = os.path.join(args.input_dir)
    args.output_first_split_dir = os.path.join(args.split_dir + "ImageSets_1st_line")
    args.output_second_split_dir = os.path.join(args.split_dir + "ImageSets_2nd_line")

    data_label_split(args)