import argparse
import cv2
import io
import numpy as np
import json
import os
import pandas as pd
import re
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def error_data_move(args):

    # img list 
    img_list = get_sub_filepaths_suffix(args.find_error_data_img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    for idx in tqdm(range(len(img_list))):

        img_name = os.path.basename(img_list[idx]).replace( '-' + (os.path.basename(img_list[idx]).split("-")[-1]), "")

        img_path = os.path.join(args.img_dir, img_name + ".jpg")
        json_path = os.path.join(args.json_dir, img_name + ".json")
        output_error_data_img_path = os.path.join(args.output_error_data_img_dir, img_name + ".jpg")
        output_error_data_json_path = os.path.join(args.output_error_data_json_dir, img_name + ".json")

        shutil.copy(img_path, output_error_data_img_path)
        shutil.copy(json_path, output_error_data_json_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.img_dir = "/yuanhuan/data/image/RM_ANPR/original/cn/china/guangzhou_8mm_2024/20231214_20240123/JPEGImages/"
    args.json_dir = "/yuanhuan/data/image/RM_ANPR/original/cn/china/guangzhou_8mm_2024/20231214_20240123/Json/"
    args.output_error_data_img_dir = "/yuanhuan/data/image/RM_ANPR/original/cn/china_error_data/guangzhou_8mm_2024_20231214_20240123/Images"
    args.output_error_data_json_dir = "/yuanhuan/data/image/RM_ANPR/original/cn/china_error_data/guangzhou_8mm_2024_20231214_20240123/Json"
    args.find_error_data_img_dir = "/yuanhuan/data/image/RM_ANPR/original/cn/china_error_data/guangzhou_8mm_2024_20231214_20240123"

    error_data_move(args)