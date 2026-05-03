#旷世res变成xml
import argparse
import cv2
import json
import numpy as np
import os
import sys 
from tqdm import tqdm
import re


sys.path.insert(0, '/yuanhuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml

def is_first_char_chinese(s):  
    if s and '\u4e00' <= s[0] <= '\u9fff':  
        return True   
    return False 

def res_to_xml(args):

    # mkdir
    os.makedirs(args.output_xml_dir, exist_ok=True)

    jpg_list = np.array(os.listdir(args.input_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    sdk_res_list = np.array(os.listdir(args.sdk_res_dir))

    for idx in tqdm(range(len(jpg_list))):
        
        jpg_name = jpg_list[idx]
        jpg_path = os.path.join(args.input_dir, jpg_name)

        # xml 
        xml_name = str(jpg_name)[:-4] + '.xml'
        xml_path = os.path.join(args.output_xml_dir, xml_name)
        
        xml_bboxes = {}
        xml_bool = False

        for idy in range(len(sdk_res_list)):

            sdk_res_name = sdk_res_list[idy]

            if jpg_name not in sdk_res_name:
                continue
            
            plate_re = re.compile(r".*plate_code\[(.*?)\]" )
            color_re = re.compile(r".*plate_color\[(.*?)\]" )
            column_re = re.compile(r".*plate_type\[(.*?)\]" )
            roi_re = re.compile(r".*rect_\[(.*?)\]" )
            confidence_re = re.compile(r".*confidence_(.*?)\.bmp" )

            # class
            plate = plate_re.findall(sdk_res_name)[0]
            if not is_first_char_chinese(plate):
                # print(plate)
                continue
            # color
            color = color_re.findall(sdk_res_name)[0]
            if color=='small_new_energy' or color=='large_new_energy':
                color='green'
                # print(color)
            if color=='absence':
                color='none'
                # print(color)
            # column
            column = column_re.findall(sdk_res_name)[0]
            column_res = "single" if column == "one_row" else "double"
            # roi
            roi = roi_re.findall(sdk_res_name)[0].split('_')
            # confidence_re
            confidence = confidence_re.findall(sdk_res_name)[0]
            if float(confidence)<args.confidence:
                # print(confidence)
                continue

            label = "license_plate_{}_{}_{}_{}".format(plate, color, column_res, confidence)
            if label not in xml_bboxes:
                xml_bboxes[label] = []              
            xml_bboxes[label].append([int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])])
            xml_bool = True

        if xml_bool:
            img = cv2.imread(jpg_path)
            img_shape = img.shape
            write_xml(xml_path, jpg_path, xml_bboxes, img_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Auto_Dinov2_Processing')
    parser.add_argument('--input_dir', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/JPEGImages_dupes_0_95", type=str)  
    parser.add_argument('--sdk_res_dir', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/plate", type=str )  
    parser.add_argument('--output_xml_dir', default="/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_kuangshi_sdk", type=str ) 
    # parser.add_argument('--confidence', default=0.7, type=float ) 
    parser.add_argument('--confidence', default=0.6, type=float ) 
    args = parser.parse_args()

    print(args.input_dir)
    print(args.sdk_res_dir)
    print(args.output_xml_dir)
    res_to_xml(args)