import argparse
import os
from re import I
import pandas as pd
import sys 
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_result_jpg(args):
    # mkdir
    create_folder(args.output_error_data_dir)

    data_pd = pd.read_csv(args.input_error_data_csv)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        
        img_path = row['file']

        for key in args.find_dict.keys():
            key_res = row[key]

            if key_res == args.find_dict[key]:

                # img
                # to_img_name = os.path.basename(img_path)
                to_img_name = os.path.basename(img_path).replace('.jpg', '_{}.jpg'.format(row['ocr']))
                # to_img_name = os.path.basename(img_path).replace('{}'.format(str(os.path.basename(img_path)).split('_')[-1]), '_{}.jpg'.format(row['ocr']))
                to_img_path = os.path.join(args.output_error_data_dir, key, to_img_name)
                create_folder(os.path.dirname(to_img_path))
                # shutil.move(img_path, to_img_path)
                shutil.copy(img_path, to_img_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/"
    
    # args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "data_us_test_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "data_us_test_result")
    # args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "analysis_data_us_trainval_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "analysis_data_us_trainval_result")
    # args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "data_us_Alabama_all_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "data_us_Alabama_all_result")
    args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "data_us_Alabama_all_threshold_result.csv")
    args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "data_us_Alabama_all_threshold_result")
    # args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "data_us_Texas_all_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "data_us_Texas_all_result")
    # args.input_error_data_csv = os.path.join(args.input_dir, "inference", "test_caffe", "data_us_Texas_all_threshold_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "inference", "error_data", "data_us_Texas_all_threshold_result")
    
    args.find_dict = {'res': 0}
    # args.find_dict = {'res': 1}
    
    find_result_jpg(args)
