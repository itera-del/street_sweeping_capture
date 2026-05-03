import argparse
import cv2
import json
import numpy as np
import os
import sys 
from tqdm import tqdm
import re


sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml


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
        img = cv2.imread(jpg_path)
        img_shape = img.shape

        # xml 
        xml_name = str(jpg_name)[:-4] + '.xml'
        xml_path = os.path.join(args.output_xml_dir, xml_name)
        xml_bboxes = {}

        for idy in range(len(sdk_res_list)):

            sdk_res_name = sdk_res_list[idy]

            if jpg_name in sdk_res_name:
                
                plate_re = re.compile(r".*plate_code\[(.*?)\]" )
                color_re = re.compile(r".*plate_color\[(.*?)\]" )
                roi_re = re.compile(r".*rect_\[(.*?)\]" )
                confidence_re = re.compile(r".*confidence_(.*?)\.bmp" )

                # class
                plate = plate_re.findall(sdk_res_name)[0]
                # color
                color = color_re.findall(sdk_res_name)[0]
                # roi
                roi = roi_re.findall(sdk_res_name)[0].split('_')
                # confidence_re
                confidence = confidence_re.findall(sdk_res_name)[0]

                label = "{}_{}_{}".format(plate, color, confidence)
                if label not in xml_bboxes:
                    xml_bboxes[label] = []              
                xml_bboxes[label].append([int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])])
                
        write_xml(xml_path, jpg_path, xml_bboxes, img_shape)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.input_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/测试视频/0619车牌数据/jpg/0000000000000000-240619-091710-091910-000005000000"
    args.sdk_res_dir = "/home/huanyuan/share/huanyuan/share/plate"
    args.output_xml_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/测试视频/0619车牌数据/xml/0000000000000000-240619-091710-091910-000005000000"

    res_to_xml(args)