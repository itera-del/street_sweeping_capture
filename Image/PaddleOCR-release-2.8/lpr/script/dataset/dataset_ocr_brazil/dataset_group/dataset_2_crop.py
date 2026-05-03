import argparse
import cv2
import io
import json
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_ocr_brazil.dataset_dict.dataset_brazil_dict import *


def dataset_group(args):
    
    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # init 
    error_list = []
    
    for idx in tqdm(range(len(img_list))):

        id_num = 0
        img_path = img_list[idx]
        json_path = img_list[idx].replace(".jpg", ".json")
        img_name = os.path.basename(img_path).replace(".jpg", "")

        # img
        try:
            img = cv2.imread(img_path)
            img_aspect = img.shape[1] / img.shape[0]
            img_column = 'Single' if img_aspect > plate_aspect else 'Double'
        except:
            print('"ori_path": {}, "type": "img"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "img"})
            continue

        # json 
        try:
            with io.open(json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()
        except:
            print('"ori_path": {}, "type": "json"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "json"})
            continue

        # shape
        if ( img_column == 'Single' and img.shape[0] < 25 ) or \
            ( img_column == 'Double' and img.shape[0] < 45 ):
            print('"ori_path": {}, "type": "img_size"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "img_size"})
            continue

        for cell in data_json['shapes']:
    
            # plate points
            if 'points' in cell:
                pts = np.array(cell["points"], np.int32)
                
                if 'type' in cell and cell["type"] == "rectangle":
                    pts = pts.reshape((-1, 1, 2))
                    try:
                        x1 = np.min((pts[0][0][0], pts[1][0][0]))
                        x2 = np.max((pts[0][0][0], pts[1][0][0]))
                        y1 = np.min((pts[0][0][1], pts[1][0][1]))
                        y2 = np.max((pts[0][0][1], pts[1][0][1]))
                        h = y2 - y1
                        w = x2 - x1

                        plate_img = img[y1:y2, x1:x2]
                    except:
                        print('"ori_path": {}, "type": "pts"'.format(img_path))
                        error_list.append({"ori_path": img_path, "type": "pts"})
                        continue

                else:
                    pts = pts.reshape((-1, 1, 2))
                    try:
                        x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                        x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                        y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                        y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                        h = y2 - y1
                        w = x2 - x1

                        plate_img = img[y1:y2, x1:x2]
                    except:
                        print('"ori_path": {}, "type": "pts"'.format(img_path))
                        error_list.append({"ori_path": img_path, "type": "pts"})
                        continue
            else:
                continue

            # plate num
            plate_num = 'none'
            if 'vehicleic' in cell:
                plate_num = cell["vehicleic"]
            elif 'attributes' in cell and len(cell["attributes"]):
                plate_num = cell["attributes"][0]["value"]
            else:
                print('"ori_path": {}, "type": "plate_num":none'.format(img_path))
                error_list.append({"ori_path": img_path, "type": "plate_num"})
                continue
            plate_num = plate_num.upper()

            # check
            if len(plate_num) == 0:
                print('"ori_path": {}, "type": "plate_num", len==0'.format(img_path))
                error_list.append({"ori_path": img_path, "type": "plate_num"})
                continue
        
            output_nmae = args.data_format.format(img_name, id_num, plate_num)
            ouuput_img_name = os.path.join(args.output_img_dir, output_nmae + '.jpg')
            id_num += 1

            # save
            create_folder(os.path.dirname(ouuput_img_name))
            if not os.path.exists(ouuput_img_name) :
                try:
                    cv2.imwrite(ouuput_img_name, plate_img)
                except:
                    print('"ori_path": {}, "type": "cv_save"'.format(img_path))
                    error_list.append({"ori_path": img_path, "type": "cv_save"})
                    continue

    # out csv
    out_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/2022_0914/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil_crop/Brazil_motor/crop_2022_0914/") 
    args = parser.parse_args()

    print("1、dataset 2 crop.")
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))
    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    args.data_format = "{}-{:0>2d}_{}"        # name-id_车牌号

    dataset_group(args)