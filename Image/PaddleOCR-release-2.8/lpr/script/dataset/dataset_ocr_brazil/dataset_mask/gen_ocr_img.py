import argparse
import shutil
import os
import pandas as pd
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def gen_ocr_img(args):

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # init 
    error_list = []

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        # plate num
        plate_num = img_name.split('-')[-1].split("_")[-1].split('.jpg')[0]
        if len(plate_num) != 7:
            print("[ERROR]: plate_num:{}, len: {}/7. ".format(plate_num, len(plate_num)))
            error_list.append({"ori_path": img_path, "type": "len_plate_num"})
            continue
        
        output_img_path = os.path.join(args.output_img_dir, img_name)

        # move
        if not os.path.exists(output_img_path):
            shutil.copy(img_path, output_img_path)

    # out csv
    out_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil_crop/Brazil_motor/crop_2022_0914/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_brazil/data_crop_2022_0914/")
    args = parser.parse_args()

    print("1、gen ocr img.")
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    gen_ocr_img(args)