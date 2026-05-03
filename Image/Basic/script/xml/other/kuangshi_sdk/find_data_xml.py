import argparse
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def find_data(args):

    # mkdir 
    os.makedirs(args.output_jpg_folder, exist_ok=True)

    # jpg_list
    # jpg_list = np.array(get_sub_filepaths_suffix(args.input_dir, suffix='.jpg'))
    jpg_list = np.array(os.listdir(args.input_dir))

    for idy in tqdm(range(len(jpg_list))):
        
        jpg_name = jpg_list[idy]
        jpg_path = os.path.join(args.input_dir, jpg_name)
        xml_path = os.path.join(args.input_xml_dir, jpg_name[:-4] + '.xml')

        # if not os.path.exists(xml_path):
        if os.path.exists(xml_path):
            output_path = os.path.join(args.output_jpg_folder, jpg_name)
            # shutil.move(jpg_path, output_path)
            shutil.copy(jpg_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/JPEGImages_dupes_0_95", type=str)  
    parser.add_argument('--input_xml_dir', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_kuangshi_sdk", type=str )  
    parser.add_argument('--output_jpg_folder', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/JPEGImages_dupes_0_95_kuangshi_sdk", type=str ) 
    args = parser.parse_args()

    find_data(args)