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


def find_error_jpg(args):
    # mkdir
    create_folder(args.output_error_data_dir)

    data_pd = pd.read_csv(args.input_error_data_csv)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        
        img_path = row['ori_path']
        error_typr = row['type']
        img_name = os.path.basename(img_path)

        # img
        to_img_path = os.path.join(args.output_error_data_dir, error_typr, img_name)
        create_folder(os.path.dirname(to_img_path))
        # shutil.move(img_path, to_img_path)
        shutil.copy(img_path, to_img_path)

        # xml
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))
        to_xml_path = os.path.join(args.output_error_data_dir, error_typr, img_name.replace(".jpg", ".xml"))
        if not os.path.exists(xml_path):
            continue
        # shutil.move(xml_path, to_xml_path)
        shutil.copy(xml_path, to_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0804_0809/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0810_0811/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0828_0831/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0901_0903/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0904_0905/"
    
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")
    args.input_error_data_csv = os.path.join(args.input_dir, "error_data", "error.csv")
    args.output_error_data_dir = os.path.join(args.input_dir, "error_data")

    find_error_jpg(args)
