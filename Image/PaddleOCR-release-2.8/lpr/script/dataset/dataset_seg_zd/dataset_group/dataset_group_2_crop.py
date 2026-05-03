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
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict import *


def dataset_group(args):
    
    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_dir, ".jpg")
    img_list.sort()
    print(len(img_list))

    # init 
    error_list = []
    error_plate_region_list = []
    error_plate_vidcolor_list = []
    error_plate_vidlan_list = []

    for idx in tqdm(range(len(img_list))):
        
        id_num = 0
        img_path = img_list[idx]
        json_path = img_list[idx].replace(".jpg", ".json")
        img_name = os.path.basename(img_path).replace(".jpg", "")
        
        # img
        try:
            img = cv2.imread(img_path)
            img_aspect = img.shape[1] / img.shape[0]
            img_column = 'Single' if img_aspect > 2.5 else 'Double'
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
                continue

            plate_num = plate_num.upper()
            for replace_name in replace_name_list:
                plate_num = plate_num.replace(replace_name, '')
                
            # check
            if len(plate_num) == 0:
                print('"ori_path": {}, "type": "plate_num", "plate_num": {}'.format(img_path, plate_num))
                error_list.append({"ori_path": img_path, "type": "plate_num"})
                continue

            bool_error_plate_num = np.array([True if str_num not in kind_num_labels else False for str_num in plate_num]).sum()
            if bool_error_plate_num:
                print('"ori_path": {}, "type": "plate_num", "plate_num": {}'.format(img_path, plate_num))
                error_list.append({"ori_path": img_path, "type": "plate_num"})
                continue  

            # plate country & city
            plate_country = country_name_list[0]
            plate_city = city_name_list[0]
            if "region" in cell:
                plate_region = cell["region"]

                if plate_region == 'AD':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[1]
                
                elif plate_region == 'Abu_Dhabi' or plate_region =='AbuDhabi':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[2]

                elif plate_region == 'Dubai' or plate_region =='DUBAI':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[3]

                elif plate_region == 'Ajman':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[4]

                elif plate_region == 'Sharjah':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[5]
                    
                elif plate_region == 'Shj':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[6]

                elif plate_region == 'Ras_Al_Khaimah' or plate_region =='RAK':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[7]

                elif plate_region == 'Umm_al_Qaiwain':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[8]

                elif plate_region == 'Fujairah':
                    plate_country = country_name_list[1]
                    plate_city = city_name_list[9]

                elif plate_region == 'unknownvehicleplate' or plate_region == 'region':
                    plate_country = country_name_list[0]
                    plate_city = city_name_list[0]
                
                else:
                    print("plate_region: {}".format(plate_region))
                    error_plate_region_list.append(plate_region)
                    # raise Exception

            # plate color
            plate_color = 'none'
            if 'vidcolor' in cell:
                plate_vidcolor = cell['vidcolor']

                if plate_vidcolor == 'red_card':
                    plate_color = 'red'
                elif plate_vidcolor == 'green_card':
                    plate_color = 'green'
                elif plate_vidcolor == 'yellow_card':
                    plate_color = 'yellow'
                elif plate_vidcolor == 'blue_card':
                    plate_color = 'blue'
                elif plate_vidcolor == 'white_card':
                    plate_color = 'white'
                elif plate_vidcolor == 'infrared_card':
                    plate_color = 'infrared'
                elif plate_vidcolor == 'black_card':
                    plate_color = 'black'
                else:
                    print("plate_vidcolor: {}".format(plate_vidcolor))
                    error_plate_vidcolor_list.append(plate_vidcolor)
                    # raise Exception

            # plate column      
            plate_column = column_name_list[0]
            if 'vidlan' in cell:
                plate_vidlan = cell['vidlan']

                if plate_vidlan == 'Double_column':
                    plate_column = column_name_list[1]
                elif plate_vidlan == 'Single_column':
                    plate_column = column_name_list[2]
                else:
                    print("plate_vidlan: {}".format(plate_vidlan))
                    error_plate_vidlan_list.append(plate_vidlan)
                    # raise Exception

            if plate_column == column_name_list[0]:
                
                if( w / h > 2.5 ):
                    plate_column = column_name_list[2]
                else:
                    plate_column = column_name_list[1]
            
            output_name = args.data_format.format(img_name, id_num, plate_country, plate_city, plate_color, plate_column, plate_num)
            ouuput_img_path = os.path.join(args.output_img_dir, output_name + '.jpg')
            id_num += 1

            # save
            create_folder(os.path.dirname(ouuput_img_path))
            if not os.path.exists(ouuput_img_path) :
                try:
                    cv2.imwrite(ouuput_img_path, plate_img)
                except:
                    print('"ori_path": {}, "type": "cv_save"'.format(img_path))
                    error_list.append({"ori_path": img_path, "type": "cv_save"})
                    continue
    
    # out csv
    out_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")

    print(error_plate_region_list)
    print(error_plate_vidcolor_list)
    print(error_plate_vidlan_list)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_1115_1116") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1115_1116/") 
    args = parser.parse_args()

    print("1、dataset group 2 crop.")
    print("input_dir: {}".format(args.input_dir))

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    args.data_format = "{}-{:0>2d}_{}_{}_{}_{}_{}"        # name-id_国家_城市_颜色_单双行_车牌号

    dataset_group(args)