import argparse
import numpy as np
import os
import shutil
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET


def copy_data(args):

    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    # image init
    img_list = np.array(os.listdir(args.jpg_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]

    for idx in tqdm(range(len(img_list))):
        jpg_path = os.path.join(args.jpg_dir, img_list[idx])
        output_jpg_path = os.path.join(args.output_dir, args.output_jpg_subdir, img_list[idx])
        os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)

        xml_path = os.path.join(args.xml_dir, img_list[idx][:-4] + ".xml")
        output_xml_path = os.path.join(args.output_dir, args.output_xml_subdir, img_list[idx][:-4] + ".xml")
        os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)

        json_path = os.path.join(args.json_dir, img_list[idx][:-4] + ".json")
        output_json_path = os.path.join(args.output_dir, args.output_json_subdir, img_list[idx][:-4] + ".json")
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)


        shutil.copy(jpg_path, output_jpg_path)
        shutil.copy(xml_path, output_xml_path)
        shutil.copy(json_path, output_json_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/JPEGImages_dupes_0_95_w_plate', type=str)  
    parser.add_argument('--xml_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS', type=str )  
    parser.add_argument('--json_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Json_w_plate', type=str ) 
    parser.add_argument('--output_dir', default='/yuanhuan/data/image/RM_ANPR/original/cn_202406/china/202407_ChongqingHunan', type=str ) 
    parser.add_argument('--output_jpg_subdir', default='JPEGImages', type=str ) 
    parser.add_argument('--output_xml_subdir', default='Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS', type=str ) 
    parser.add_argument('--output_json_subdir', default='Json', type=str ) 
    args = parser.parse_args()

    copy_data(args)