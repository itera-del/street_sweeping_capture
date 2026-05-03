import argparse
import importlib
import numpy as np
import os 
import pandas as pd
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def analysis_dataset_label(args):
    
    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.analysis_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # init
    analysis_dict = {}

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))

        # load object_roi_list
        object_roi_list = dataset_dict.load_ori_object_roi(xml_path)

        for key_name in dataset_dict.analysis_label_columns.keys():
    
            for idz in range(len(object_roi_list)):
                classname = object_roi_list[idz]["classname"]
                bndbox = object_roi_list[idz]["bndbox"]

                if classname not in dataset_dict.analysis_label_columns[key_name]:
                    continue

                if key_name not in analysis_dict:
                    analysis_dict[key_name] = {}
                
                if classname not in analysis_dict[key_name]:
                    analysis_dict[key_name][classname] = 1
                else:
                    analysis_dict[key_name][classname] += 1
    
    for key_name in dataset_dict.analysis_label_columns.keys():

        if key_name not in analysis_dict:
            continue

        out_csv_path = os.path.join(args.analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name], index=[0], columns=dataset_dict.analysis_label_columns[key_name])
        analysis_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


def analysis_dataset_label_from_name(args):
 
    # mkdir
    create_folder(args.analysis_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # init
    analysis_dict = {}

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        img_label_list = str(img_name).split('.jpg')[0].split('-')[-1].split('_')

        for label_idx in range(len(img_label_list)):

            classname = img_label_list[label_idx]

            # country
            if label_idx == 1:
                key_name = 'country'
            elif label_idx == 2:
                key_name = 'city'
            elif label_idx == 3:
                key_name = 'color'
            else:
                continue

            if key_name not in analysis_dict:
                analysis_dict[key_name] = {}

            if classname not in analysis_dict[key_name]:
                analysis_dict[key_name][classname] = 1
            else:
                analysis_dict[key_name][classname] += 1
    
    for key_name in analysis_dict:

        out_csv_path = os.path.join(args.analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name], index=[0])
        analysis_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/") 
    args = parser.parse_args()

    print("1、analysis dataset seg mask label num.")
    print("input_dir: {}".format(args.input_dir))

    ###############################################
    ## from xml
    ###############################################
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")
    args.analysis_dir = os.path.join(args.input_dir, "analysis")

    args.seg_dict_name = "dataset_zd_dict_nomal"
    analysis_dataset_label(args)