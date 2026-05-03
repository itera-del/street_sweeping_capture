import argparse
import cv2
import os
import pandas as pd
import sys
from tqdm import tqdm
import shutil


def dataset_copy(args):

    # mkdir
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_json_dir, exist_ok=True)

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
    
        # info
        img_path = row['img_path']
        if not os.path.exists(img_path):
            img_path = os.path.join(os.path.dirname(img_path), "JPEGImages", os.path.basename(img_path))
            assert os.path.exists(img_path)
        json_path = row['json_path']
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(json_path), "Json", os.path.basename(json_path))
            assert os.path.exists(json_path)
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color']
        plate_column = row['column']
        plate_num = row['num']

        to_img_path = os.path.join(args.output_image_dir, os.path.basename(img_path))
        out_json_path = os.path.join(args.output_json_dir, os.path.basename(json_path))
        if not os.path.exists(to_img_path) and not os.path.exists(out_json_path):
            shutil.copy(img_path, to_img_path)
            shutil.copy(json_path, out_json_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Shanxi") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/cn_2/temp") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/cn_202406/china") 
    args = parser.parse_args()

    args.input_csv_path = os.path.join(args.input_dir, args.date_name + '.csv')
    args.output_image_dir = os.path.join(args.output_dir, args.date_name, "JPEGImages")
    args.output_json_dir = os.path.join(args.output_dir, args.date_name, "Json")
    dataset_copy(args)