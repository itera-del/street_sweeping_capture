import argparse
import shutil
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
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict import *


def seg_check(args):
    
    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # init 
    error_list = []

    for idx in tqdm(range(len(img_list))):
        
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))

        # plate_num
        plate_num = img_name.split('.jpg')[0].split('_')[-1]
        plate_num = plate_num.upper()
        for replace_name in replace_name_list:
            plate_num = plate_num.replace(replace_name, '')

        if plate_num == '':
            # 标签存在问题，没有车牌号标注
            print('"ori_path": {}, "type": "plate_num"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "plate_num"})
            continue

        # info
        country_list = []
        city_list = []
        car_type_list = []
        country_f_list = []
        city_f_list = []
        car_type_f_list = []
        color_list = []
        
        # xml
        try:
            tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
            root = tree.getroot()   # 获取根节点
        except:
            print('"ori_path": {}, "type": "xml"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "xml"})
            continue

        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            if classname in ignore_unknown:
                continue

            if classname in change_name_dict:
                classname = change_name_dict[classname]
            
            if classname in kind_name_list or \
                classname in num_name_list or \
                classname in country_name_list or \
                classname in city_name_list or \
                classname in car_type_name_list or \
                classname in country_f_name_list or \
                classname in city_f_name_list or \
                classname in car_type_f_name_list or \
                classname in color_name_list:

                if classname in country_name_list:
                    country_list.append(classname) 
                
                if classname in city_name_list:
                    city_list.append(classname)

                if classname in car_type_name_list:
                    car_type_list.append(classname)

                if classname in country_f_name_list:
                    country_f_list.append(classname)

                if classname in city_f_name_list:
                    city_f_list.append(classname)

                if classname in car_type_f_name_list:
                    car_type_f_list.append(classname)

                if classname in color_name_list:
                    color_list.append(classname)
            else:
                print(classname)
                raise Exception

        country_list = list(set(country_list))
        city_list = list(set(city_list))
        car_type_list = list(set(car_type_list))

        country_f_list = list(set(country_f_list))
        city_f_list = list(set(city_f_list))
        car_type_f_list = list(set(car_type_f_list))
        color_list = list(set(color_list))

        if len(country_list) > 1:
            # 标签存在问题，多个国家
            print('"ori_path": {}, "type": "country"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "country"})
        else:
            pass

        if len(city_list) > 1:
            # 标签存在问题，多个城市
            print('"ori_path": {}, "type": "city"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "city"})
        else:
            pass

        if len(car_type_list) > 1:
            # 标签存在问题，多个车型
            print('"ori_path": {}, "type": "car_type"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "car_type"})
        else:
            pass

        if len(country_f_list) > 1:
            # 标签存在问题，多个车型
            print('"ori_path": {}, "type": "country_f"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "country_f"})
        else:
            pass

        if len(city_f_list) > 1:
            # 标签存在问题，多个车型
            print('"ori_path": {}, "type": "city_f"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "city_f"})
        else:
            pass

        if len(car_type_f_list) > 1:
            # 标签存在问题，多个车型
            print('"ori_path": {}, "type": "car_type_f"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "car_type_f"})
        else:
            pass

        if len(color_list) > 1:
            # 标签存在问题，多个车型
            print('"ori_path": {}, "type": "color"'.format(img_path))
            error_list.append({"ori_path": img_path, "type": "color"})
        else:
            pass

        output_img_name = os.path.join(args.output_img_dir, img_name)
        output_xml_name = os.path.join(args.output_xml_dir, img_name.replace(".jpg", ".xml"))

        # move
        if not os.path.exists(output_img_name) and not os.path.exists(output_xml_name):
            shutil.copy(img_path, output_img_name)
            shutil.copy(xml_path, output_xml_name)
    
    # out csv
    out_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1115_1116/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/")
    args = parser.parse_args()

    print("2、dataset group check.")
    print("input_dir: {}".format(args.input_dir))

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")
    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_xml_dir = os.path.join(args.output_dir, "xml")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    seg_check(args)