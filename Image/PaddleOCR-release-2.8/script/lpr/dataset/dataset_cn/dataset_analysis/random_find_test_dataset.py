import argparse
import cv2
import os
import random
import pandas as pd
import sys
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal import *


def random_find_test_dataset(args):

    # mkdir
    create_folder(args.output_test_data_dir)
    create_folder(args.output_test_data_JPEGImages_dir)
    create_folder(args.output_test_data_Json_dir)

    # init
    analysis_dict = {}

    # pd
    data_pd = pd.read_csv(args.input_csv_path)
    
    # scale
    random_scale = float(args.data_num) / float(len(data_pd))

    # data_num
    data_num = 0
    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        if random.random() > random_scale:
            continue
        
        if data_num >= args.data_num:
            break

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color']
        plate_column = row['column']
        plate_num = row['num']

        if "JPEGImages" not in img_path:
            img_path = os.path.join(os.path.dirname(img_path), "JPEGImages", os.path.basename(img_path))
            json_path = os.path.join(os.path.dirname(json_path), "Json", os.path.basename(json_path))
        img_name = os.path.basename(img_path)
        json_name = os.path.basename(json_path)
        output_img_name = os.path.join(args.output_test_data_JPEGImages_dir, img_name)
        output_json_name = os.path.join(args.output_test_data_Json_dir, json_name)
        shutil.copy(img_path, output_img_name)
        shutil.copy(json_path, output_json_name)
        
        data_num += 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="beijing") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/cn/china_csv/") 
    parser.add_argument('--output_test_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/cn/china_analysis_test_data/") 
    parser.add_argument('--data_num', type=int, default=20) 
    args = parser.parse_args()

    if "/" not in args.date_name:
        date_name = str(args.date_name)
    else:
        date_name = '_'.join(str(args.date_name).split('/'))
    args.input_csv_path = os.path.join(args.input_csv_dir, date_name + '.csv')
    args.output_test_data_JPEGImages_dir = os.path.join(args.output_test_data_dir, "JPEGImages")
    args.output_test_data_Json_dir = os.path.join(args.output_test_data_dir, "Json")

    print("random find test dataset.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))

    random_find_test_dataset(args)