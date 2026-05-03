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
                to_img_path = os.path.join(args.output_error_data_dir, key, to_img_name)
                create_folder(os.path.dirname(to_img_path))
                # shutil.move(img_path, to_img_path)
                shutil.copy(img_path, to_img_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_hisi_1010/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1111/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1120/"
    # args.input_dir = "/yuanhuan/code/demo/Image/recognition2d/PaddleOCR/output/"
    args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/best_accuracy/"
    
    # ocr_merge_test
    args.input_error_data_csv = os.path.join(args.input_dir, "test", "ocr_merge_test_result.csv")
    args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "ocr_merge_test")
    # args.input_error_data_csv = os.path.join(args.input_dir, "trainval", "ocr_merge_test_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "trainval", "error_data", "ocr_merge_test")

    # data_crop
    # args.input_error_data_csv = os.path.join(args.input_dir, "test", "data_crop_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "data_crop")

    # from_jpg_dir
    # args.input_error_data_csv = os.path.join(args.input_dir, 'test/data_synthesis_X_result.csv')
    # args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "data_kind", "X")
    # args.input_error_data_csv = os.path.join(args.input_dir, 'test/data_synthesis_Y_result.csv')
    # args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "data_kind", "Y")
    # args.input_error_data_csv = os.path.join(args.input_dir, 'test/data_synthesis_Z_result.csv')
    # args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "data_kind", "Z")
    
    args.find_dict = {'res': 0}
    
    find_result_jpg(args)
