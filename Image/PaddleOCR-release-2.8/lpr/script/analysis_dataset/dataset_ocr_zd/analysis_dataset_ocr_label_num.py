import argparse
import importlib
import os 
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def analysis_dataset_label(args):

    # mkdir
    create_folder(args.analysis_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # init
    analysis_dict = {}

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        kind_name = img_name.replace('.jpg', '').split('_')[-1].split('#')[0]

        if kind_name not in analysis_dict:
            analysis_dict[kind_name] = 1
        
        analysis_dict[kind_name] += 1
        
    analysis_list = list(analysis_dict.keys())
    analysis_list.sort()
    
    out_csv_path = os.path.join(args.analysis_dir, 'analysis_{}.csv'.format("kind"))
    analysis_pd = pd.DataFrame(analysis_dict, index=[0], columns=analysis_list)
    analysis_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


def analysis_dataset_label_all(args):
    
    # mkdir
    create_folder(args.analysis_dir)
    
    img_list = []
    for idx in range(len(args.input_dir_list)):
        img_list.extend(get_sub_filepaths_suffix(os.path.join(args.input_dir_list[idx],args. input_img_dir), ".jpg"))
    img_list.sort()

    # init
    analysis_dict = {}

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        kind_name = img_name.replace('.jpg', '').split('_')[-1].split('#')[0]

        if kind_name not in analysis_dict:
            analysis_dict[kind_name] = 1
        
        analysis_dict[kind_name] += 1
        
    analysis_list = list(analysis_dict.keys())
    analysis_list.sort()
    
    out_csv_path = os.path.join(args.analysis_dir, 'analysis_{}.csv'.format("kind"))
    analysis_pd = pd.DataFrame(analysis_dict, index=[0], columns=analysis_list)
    analysis_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_1115_1116/") 
    args = parser.parse_args()

    print("1、analysis dataset ocr label num.")
    print("input_dir: {}".format(args.input_dir))

    ##############################################
    # normal
    ##############################################

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.analysis_dir = os.path.join(args.input_dir, "analysis")

    analysis_dataset_label(args)

    # ###############################################
    # ## all
    # ###############################################
    # args.input_dir_list = [
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop/",
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_0804_0809/",
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_0810_0811/",
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_0828_0831/",
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_0901_0903/",
    #     "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_0904_0905/",
    # ]

    # args.input_img_dir = "Images"
    # args.analysis_dir = os.path.join("/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/", "analysis")

    # analysis_dataset_label_all(args)