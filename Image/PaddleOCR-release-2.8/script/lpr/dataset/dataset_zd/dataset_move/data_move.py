import argparse
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm

def data_move(args):

    # mkdir 
    os.makedirs(args.output_jpg_dir, exist_ok=True)
    os.makedirs(args.output_xml_dir, exist_ok=True)
    
    # jpg_list
    jpg_list = np.array(os.listdir(args.jpg_dir))

    for idy in tqdm(range(len(jpg_list))):
        
        jpg_name = jpg_list[idy]
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        output_img_path = os.path.join(args.output_jpg_dir, jpg_name)
        xml_name = jpg_name[:-4] + '.xml'
        xml_path = os.path.join(args.xml_dir, xml_name)
        output_xml_path = os.path.join(args.output_xml_dir, xml_name)
        
        if not jpg_name.endswith(args.jpg_suffix) or \
            not jpg_name.endswith(args.jpg_suffix) or \
            not os.path.exists(jpg_path) or \
            not os.path.exists(xml_path):
            print(jpg_path)
            print(xml_path)
            continue
        
        shutil.copy(jpg_path, output_img_path)
        shutil.copy(xml_path, output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/dataset_closed_loop/frames/capture/capture_zd_20241108_frames_5f/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/uae_20241108/") 
    parser.add_argument("--jpg_subdir", type=str, default="batch_1_select_manual")
    parser.add_argument("--xml_subdir", type=str, default="xml_capture_det_v1.0_1")
    parser.add_argument("--output_jpg_subdir", type=str, default="JPEGImages")
    parser.add_argument("--output_xml_subdir", type=str, default="Annotations")

    args = parser.parse_args()

    args.jpg_dir = os.path.join(args.input_dir, args.jpg_subdir)
    args.xml_dir = os.path.join(args.input_dir, args.xml_subdir)
    args.output_jpg_dir = os.path.join(args.output_dir, args.output_jpg_subdir)
    args.output_xml_dir = os.path.join(args.output_dir, args.output_xml_subdir)

    args.jpg_suffix = ".jpg"
    args.xml_suffix = ".xml"

    data_move(args)