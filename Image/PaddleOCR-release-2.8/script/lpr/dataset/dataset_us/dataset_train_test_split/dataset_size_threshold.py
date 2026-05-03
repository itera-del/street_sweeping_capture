import argparse
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    
    # mkdir
    create_folder(os.path.dirname(args.all_file))

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    # file_list
    with open(args.all_file, "w") as f:
        for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            
            # info
            plate_num = row['num']
            file_path = row["roi_img_path"]
            plate_roi = row['roi']
            plate_roi_list = plate_roi.split(',')
            x1 = int(plate_roi_list[0])
            x2 = int(plate_roi_list[1])
            y1 = int(plate_roi_list[2])
            y2 = int(plate_roi_list[3])

            width = x2 - x1
            height = y2 - y1

            if width > args.width_threshold and height > args.height_threshold:
                f.write('{}'.format(file_path))
                f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Alabama_2022") 
    parser.add_argument('--ocr_name', type=str, default="plate_us_202406") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("data train test split.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    args.width_threshold = 0
    args.height_threshold = 60

    args.input_csv_path = os.path.join(args.input_dir, '{}.csv'.format(args.date_name))
    args.all_file = os.path.join(args.input_dir, "ImageSets/Main/all_threshold.txt")

    split(args)