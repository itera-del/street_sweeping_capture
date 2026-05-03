import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.script.xml.xml_write import write_xml


def intersect(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y1 = max(box_a[1], box_b[1])
    inter_y2 = min(box_a[3], box_b[3])
    inter =  max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0) 
    return inter


def bool_box_in_roi(box, roi):
    bool_in_w = True if box[0] >= roi[0] and box[2] <= roi[2] else False
    bool_in_h = True if box[1] >= roi[1] and box[3] <= roi[3] else False
    return bool_in_w * bool_in_h


def match_bbox_iou(input_roi, match_roi_list):
    # init
    matched_idx_list = []
    max_intersect_iou = 0.0
    max_intersect_iou_idx = 0

    for idx in range(len(match_roi_list)):
        match_roi_idx = match_roi_list[idx][0:4]
        intersect_iou = intersect(input_roi, match_roi_idx)

        if intersect_iou > max_intersect_iou:
            max_intersect_iou = intersect_iou
            max_intersect_iou_idx = idx
        
    if max_intersect_iou > 0.0:
        matched_idx_list.append(max_intersect_iou_idx)
    
    return matched_idx_list, max_intersect_iou


def xml_change(args):

    # mkdir 
    create_folder(args.output_xml_dir)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith(args.suffix) for jpg in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
    
        img_name = img_list[idx]
        xml_name = img_name.replace(args.suffix, ".xml")
        img_path = os.path.join(args.img_dir, img_name)
        xml_path = os.path.join(args.xml_dir, xml_name)
        xml_2_path = os.path.join(args.xml_2_dir, xml_name)
        output_xml_path = os.path.join(args.output_xml_dir, xml_name)

        # check
        if not (os.path.exists(xml_path) and os.path.exists(xml_2_path)):
            # print(img_path)
            # print(xml_path)
            # print(xml_2_path)

            continue

        # img
        img = cv2.imread(img_path)     

        # show_bboxes
        show_bboxes = {}

        # read xml
        tree_1 = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root_1 = tree_1.getroot()   # 获取根节点
        tree_2 = ET.parse(xml_2_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root_2 = tree_2.getroot()   # 获取根节点
        
        find_bool = False
        for object_1 in root_1.findall('object'):
            # name
            classname_1 = str(object_1.find('name').text)

            # bbox
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bbox_1 = object_1.find('bndbox')
            bndbox_1 = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox_1.find(pt).text))
                bndbox_1.append(cur_pt)

            bbox_2_list = []
            for object_2 in root_2.findall('object'):

                # name
                classname_2 = str(object_2.find('name').text)

                # bbox
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bbox_2 = object_2.find('bndbox')
                bndbox_2 = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox_2.find(pt).text))
                    bndbox_2.append(cur_pt)

                if classname_2 == "license":
                    bbox_2_list.append(bndbox_2)

            match_object_idx_list, iou = match_bbox_iou(bndbox_1, bbox_2_list)
            if len(match_object_idx_list):
                find_bool = True
                if classname_1 not in show_bboxes:
                    show_bboxes[classname_1] = []
                show_bboxes[classname_1].append(bbox_2_list[match_object_idx_list[0]])
            else:
                pass

        # save xml
        if find_bool:
            write_xml(output_xml_path, img_path, show_bboxes, img.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/JPEGImages_dupes_0_95', type=str )  
    parser.add_argument('--xml_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_kuangshi_sdk', type=str ) 
    parser.add_argument('--xml_2_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS', type=str) 
    parser.add_argument('--output_xml_dir', default='/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_5/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS_w_plate', type=str) 
    parser.add_argument('--suffix', default='.jpg') 
    args = parser.parse_args()

    xml_change(args)