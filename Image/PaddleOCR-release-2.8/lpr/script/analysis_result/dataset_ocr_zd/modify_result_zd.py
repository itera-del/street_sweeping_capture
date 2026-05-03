import argparse
import importlib
import io
import json
import os
import shutil
import sys

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def modify_plate_name(data_json, find_plate_num, to_plate_num, dataset_dict):
    
    bool_modify = False
    for cell in data_json['shapes']:

        # plate num
        plate_num = 'none'
        if 'vehicleic' in cell:
            plate_num = cell["vehicleic"]
        elif 'attributes' in cell and len(cell["attributes"]):
            plate_num = cell["attributes"][0]["value"]

        plate_num = plate_num.upper()
        for replace_name in dataset_dict.replace_name_list:
            plate_num = plate_num.replace(replace_name, '')

        if plate_num == find_plate_num:
            
            if 'vehicleic' in cell:
                cell["vehicleic"] = to_plate_num
            elif 'attributes' in cell and len(cell["attributes"]):
                cell["attributes"][0]["value"] = to_plate_num
            
            if bool_modify == False:
                bool_modify = True
            elif bool_modify == True:
                bool_modify = False
    
    return bool_modify


def modify_UAE(args):

    # dataset_zd_dict
    # city
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.dict_name) 

    # img list 
    img_list = []   
    for idx in range(len(args.input_dir_list)):
        img_list.extend(get_sub_filepaths_suffix(args.input_dir_list[idx], ".jpg"))
    img_list.sort()
    # print("len(img_list): {}".format(len(img_list)))

    # find list 
    find_list = []
    for idx in range(len(img_list)):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        if str(img_name).startswith(args.replace_img_name):
            find_list.append(img_path)
    # print("len(find_list): {}".format(len(find_list)))
    assert len(find_list) == 1

    img_path = find_list[0]
    json_path = find_list[0].replace(".jpg", ".json")
    print(img_path)
    print(json_path)

    # json 
    with io.open(json_path, "r", encoding="UTF-8") as f:
        data_json = json.load(f, encoding='utf-8')
        f.close()

    modify_bool = modify_plate_name(data_json, args.replace_plate_name, args.to_plate_name, dataset_dict)
    assert modify_bool == True
    print("{} -> {}".format(args.replace_plate_name, args.to_plate_name))

    with open(json_path, "w") as f:
        f.write(json.dumps(data_json, ensure_ascii=False, indent=4, separators=(',', ':')))


def modify_UAE_crop(args):

    # img list 
    img_list = []   
    for idx in range(len(args.input_dir_list)):
        img_list.extend(get_sub_filepaths_suffix(args.input_dir_list[idx], ".jpg"))
    img_list.sort()
    # print("len(img_list): {}".format(len(img_list)))

    # find list 
    find_list = []
    for idx in range(len(img_list)):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        if str(img_name).startswith(args.replace_img_name) and \
            args.replace_plate_name in str(img_name):
            find_list.append(img_path)
    # print("len(find_list): {}".format(len(find_list)))
    assert len(find_list) == 1

    img_path = find_list[0]
    json_path = find_list[0].replace(args.input_img_dir, args.input_json_dir).replace(".jpg", ".json")
    xml_path = find_list[0].replace(args.input_img_dir, args.input_xml_dir).replace(".jpg", ".xml")
    assert os.path.exists(json_path)
    assert os.path.exists(xml_path)
    print(img_path)
    print(json_path)
    print(xml_path)

    output_img_path = img_path.replace(args.replace_plate_name, args.to_plate_name)
    output_json_path = json_path.replace(args.replace_plate_name, args.to_plate_name)
    output_xml_path = xml_path.replace(args.replace_plate_name, args.to_plate_name)
    print(output_img_path)
    print(output_json_path)
    print(output_xml_path)
    shutil.move(img_path, output_img_path)
    shutil.move(json_path, output_json_path)
    shutil.move(xml_path, output_xml_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.replace_img_name = "0000000000000000-220804-131737-131833-00000D000150_26.56_43.63-sn00070"
    # args.replace_plate_name = "J#1886"
    # args.to_plate_name = "J#18886"
    args.replace_img_name = "0000000000000000-220830-145600-151735-00000D000200_18_96_25_35_00090"
    args.replace_plate_name = "M#4710"
    args.to_plate_name = "4710"
    
    args.dict_name = "dataset_zd_dict_nomal"
    args.input_dir_list = [
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0804_0809/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0810_0811/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0828_0831/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0901_0903/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0904_0905/",
    ]

    modify_UAE(args)

    args.input_dir_list = [
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0804_0809/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0810_0811/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0828_0831/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0901_0903/",
        "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0904_0905/",
    ]
    args.input_img_dir = "Images"
    args.input_json_dir = "Json"
    args.input_xml_dir = "xml"

    modify_UAE_crop(args)