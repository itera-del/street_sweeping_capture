import argparse
import io
import json
import os
import pandas as pd
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_morocco.dataset_dict.dataset_morocco_dict_normal import *

def get_kind_num_id(kind_num_name):
    # init
    kind_id, num_id, city_id  = '', '', ''

    img_name_list = kind_num_name.split('#')
    if len(img_name_list) == 1:
        num_id = img_name_list[0]
    elif len(img_name_list) == 2:
        city_id = img_name_list[0]
        num_id = img_name_list[1]
    elif len(img_name_list) == 3:
        kind_id = img_name_list[0]
        city_id = img_name_list[1]
        num_id = img_name_list[2]

    return kind_id, num_id, city_id


def dataset_csv(args):

    # mkdir
    create_folder(args.output_csv_dir)
    create_folder(args.output_error_crop_data_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "num": "", "crop_img": "", "crop_xml": "", "crop_json": ""}]
    error_list = []         # [{"img_path": "", "json_path": "", "crop_img": "", "crop_xml": "", "crop_json": "", "type": "", "value": ""}]

    # pd
    data_pd = pd.read_csv(args.input_csv_path) 

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id']
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color'] 
        plate_column = row['column'] 
        plate_num = row['num'] 
        crop_img_path = os.path.join(args.img_dir, plate_name + ".jpg")
        crop_xml_path = os.path.join(args.xml_dir, plate_name + ".xml")
        crop_json_path = os.path.join(args.json_dir, plate_name + ".json")
    
        # kind_id & num_id & city_id
        kind_num_name = plate_num
        kind_id, num_id, city_id = get_kind_num_id(kind_num_name)

        # init 
        kind_list = []
        num_list = []
        city_list = []
        # json 
        try:
            with open(crop_json_path, 'r', encoding='UTF-8') as fr:
                data_json = json.load(fr)
        except:
            print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "json"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
            error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "json", "value": ""})
            continue
        
        for cell in data_json['shapes']:

            if "license_kind" == cell["label"]:
                kind_list.append(cell)

            elif "license_num" == cell["label"]:
                num_list.append(cell)

            elif "license_city" == cell["label"]:
                city_list.append(cell)

        kind_error_bool = False
        if kind_id == "":
            if len(kind_list) != 0:
                kind_error_bool = True
        else:
            if len(kind_list) != 1:
                kind_error_bool = True
            elif kind_list[0]["attributes"][0]["value"] != kind_id:
                kind_error_bool = True

        num_error_bool = False
        if num_id == "":
            if len(num_list) != 0:
                num_error_bool = True
        else:
            if len(num_list) != 1:
                num_error_bool = True
            elif num_list[0]["attributes"][0]["value"] != "num" :
                if num_list[0]["attributes"][0]["value"] != num_id:
                    num_error_bool = True

        city_error_bool = False
        if city_id == "":
            if len(city_list) != 0:
                city_error_bool = True
        else:
            if len(city_list) != 1:
                city_error_bool = True

        if kind_error_bool:
            # 标签存在问题，kind
            print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "kind"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
            error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "kind", "value": ""})
            continue

        if num_error_bool:
            # 标签存在问题，kind
            print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "kind"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
            error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "kind", "value": ""})
            continue

        if city_error_bool:
            # 标签存在问题，kind
            print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "kind"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
            error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "kind", "value": ""})
            continue
        
        csv_list.append({"img_path": img_path, "json_path": json_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_data_csv_path = os.path.join(args.output_error_crop_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(error_data_csv_path, index=False, encoding="utf_8_sig")


def write_error_data(args):
    
    if not args.bool_write_error_data:
        return

    create_folder(args.output_error_data_img_dir)
    create_folder(args.output_error_data_xml_dir)
    create_folder(args.output_error_data_json_dir)
    create_folder(args.output_error_data_ori_img_dir)
    create_folder(args.output_error_data_ori_json_dir)

    # pd
    error_data_csv_path = os.path.join(args.output_error_crop_data_dir, 'error.csv')

    try:
        data_pd = pd.read_csv(error_data_csv_path)
    except:
        return
    
    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        crop_img_path = row['crop_img']
        crop_xml_path = row['crop_xml']
        crop_json_path = row['crop_json']
        img_name = os.path.basename(crop_img_path)
        xml_name = os.path.basename(crop_xml_path)
        json_name = os.path.basename(crop_json_path)
        ori_img_name = os.path.basename(img_path)
        ori_json_name = os.path.basename(json_path)

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_img_dir, img_name)
            shutil.copy(crop_img_path, to_img_path)
        except:
            pass
        
        # xml
        try:
            to_xml_path = os.path.join(args.output_error_data_xml_dir, xml_name)
            shutil.copy(crop_xml_path, to_xml_path)
        except:
            pass

        # json
        try:
            to_json_path = os.path.join(args.output_error_data_json_dir, json_name)
            shutil.copy(crop_json_path, to_json_path)
        except:
            pass

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_ori_img_dir, ori_img_name)
            shutil.copy(img_path, to_img_path)
        except:
            pass

        # json
        try:
            to_json_path = os.path.join(args.output_error_data_ori_json_dir, ori_json_name)
            shutil.copy(json_path, to_json_path)
        except:
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="morocco_20250329") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco_csv/") 
    parser.add_argument('--input_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco_crop/") 
    parser.add_argument('--output_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco_crop_csv/") 
    parser.add_argument('--output_error_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco_error_crop_data/")  
    parser.add_argument('--bool_write_error_data', action='store_true', default=False) 
    parser.add_argument('--img_folder', type=str, default="Images") 
    parser.add_argument('--json_folder', type=str, default="Json") 
    parser.add_argument('--xml_folder', type=str, default="xml") 
    
    args = parser.parse_args()
    args.bool_write_error_data = True

    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.img_dir = os.path.join(args.input_crop_data_dir, args.date_name, args.img_folder)
    args.xml_dir = os.path.join(args.input_crop_data_dir, args.date_name, args.xml_folder)
    args.json_dir = os.path.join(args.input_crop_data_dir, args.date_name, args.json_folder)

    args.output_csv_path = os.path.join(args.output_csv_dir, args.date_name + '.csv')
    args.output_error_crop_data_dir = os.path.join(args.output_error_crop_data_dir, args.date_name)
    args.output_error_data_img_dir = os.path.join(args.output_error_crop_data_dir, args.img_folder)
    args.output_error_data_xml_dir = os.path.join(args.output_error_crop_data_dir, args.xml_folder)
    args.output_error_data_json_dir = os.path.join(args.output_error_crop_data_dir, args.json_folder)

    args.output_error_data_ori_img_dir = os.path.join(args.output_error_crop_data_dir, "ori_img", args.img_folder)
    args.output_error_data_ori_json_dir = os.path.join(args.output_error_crop_data_dir, "ori_img", args.json_folder)

    print("dataset crop csv.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_csv_path: {}".format(args.output_csv_path))

    # 生成 dataset csv
    dataset_csv(args)

    # 保存 error data
    write_error_data(args)