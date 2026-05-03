import argparse
import numpy as np
import os 
import sys
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def dataset_type_find(args):
    # mkdir
    create_folder(args.type_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_img_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for jpg_idx in tqdm(range(len(jpg_list))):
        jpg_name = jpg_list[jpg_idx]
        jpg_path = os.path.join(args.input_img_dir, jpg_name)
        xml_path = os.path.join(args.input_img_dir, jpg_name.replace(".jpg", ".xml"))
        jpg_name_list = str(jpg_name).split('_')

        for key_name in args.type_find_dict.keys():
            key_id = args.type_find_dict[key_name]
            key_res = jpg_name_list[key_id]

            if key_res != 'none':
                output_jpg_path = os.path.join(args.type_dir, key_name, key_res, jpg_name)
                output_xml_path = os.path.join(args.type_dir, key_name, key_res, jpg_name.replace(".jpg", ".xml"))
                
                # mkdir
                create_folder(os.path.dirname(output_jpg_path))

                shutil.copy(jpg_path, output_jpg_path)
                shutil.copy(xml_path, output_xml_path)


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_city_cartype_kind_num_zd/"

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.type_dir = os.path.join(args.input_dir, "type")

    args.type_find_dict = {
                                'city': 3, 
                                'car_type': 4
                                }

    dataset_type_find(args)


if __name__ == "__main__":
    main()