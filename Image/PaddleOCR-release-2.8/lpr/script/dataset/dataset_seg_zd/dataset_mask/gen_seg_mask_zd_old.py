import argparse
import cv2
import importlib
import numpy as np
import os
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def load_kind_num_bbox(xml_path, dataset_dict):
    
    # kind & num dict
    kind_num_bbox_dict = {}

    # xml
    tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
    root = tree.getroot()   # 获取根节点

    for object in root.findall('object'):
        # name
        classname = str(object.find('name').text)

        # bbox
        bbox = object.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(float(bbox.find(pt).text))
            bndbox.append(cur_pt)
    
        if classname in dataset_dict.kind_num_labels:
            if classname not in kind_num_bbox_dict:
                kind_num_bbox_dict[classname] = []

            kind_num_bbox_dict[classname].append(bndbox)

    return kind_num_bbox_dict


def get_kind_num_id(kind_num_name):
    # init
    kind_id, num_id  = '', ''

    img_name_list = kind_num_name.split('#')
    if len(img_name_list) == 1:
        num_id = img_name_list[0]
    elif len(img_name_list) == 2:
        if len(img_name_list[0]) < len(img_name_list[1]):
            kind_id = img_name_list[0]
            num_id = img_name_list[1]
        else:
            kind_id = img_name_list[1]
            num_id = img_name_list[0]

    return kind_id, num_id


def cal_cost_dis(bbox_idx, bbox_dict):
    # init
    cost = 0.0

    bbox_idx_center_x = (bbox_idx[0] + bbox_idx[2]) / 2.0
    bbox_idx_center_y = (bbox_idx[1] + bbox_idx[3]) / 2.0

    for key in bbox_dict.keys():
        for idy in range(len(bbox_dict[key])):
            bbox_idy = bbox_dict[key][idy]
            bbox_idy_center_x = (bbox_idy[0] + bbox_idy[2]) / 2.0
            bbox_idy_center_y = (bbox_idy[1] + bbox_idy[3]) / 2.0

            cost += pow(abs(bbox_idx_center_x - bbox_idy_center_x), 2) + pow(abs(bbox_idx_center_y - bbox_idy_center_y), 2)

    return cost


def get_kind_num_bbox(kind_id, num_id, kind_num_bbox_dict):
    # init
    kind_bbox, num_bbox = [], []
    kind_bbox_list, num_bbox_list = [], []
    kind_num_bool = False

    # init kind_num_bbox_dict
    for key in kind_num_bbox_dict.keys():
        for idx in range(len(kind_num_bbox_dict[key])):
            kind_num_bbox_idy = kind_num_bbox_dict[key][idx]
            cost = cal_cost_dis(kind_num_bbox_idy, kind_num_bbox_dict)
            find_bool = False
            kind_num_bbox_idy.append(cost)
            kind_num_bbox_idy.append(find_bool)

    # find kind_bbox_list & num_bbox_list
    for idx in range(len(kind_id)):
        kind_id_idx = kind_id[idx]
        if kind_id_idx in kind_num_bbox_dict:
            kind_num_bbox_idx = kind_num_bbox_dict[kind_id_idx]
            kind_num_bbox_idx = sorted(kind_num_bbox_idx, key=lambda x: x[4], reverse=True)

            for idy in range(len(kind_num_bbox_idx)):
                kind_num_bbox_idy = kind_num_bbox_idx[idy]
                if kind_num_bbox_idy[-1] == False:
                    kind_bbox_list.append(kind_num_bbox_idy)
                    find_bool = True
                    kind_num_bbox_idy[-1] = find_bool
                    break

    for idx in range(len(num_id)):
        num_id_idx = num_id[idx]
        if num_id_idx in kind_num_bbox_dict:
            kind_num_bbox_idx = kind_num_bbox_dict[num_id_idx]
            kind_num_bbox_idx = sorted(kind_num_bbox_idx, key=lambda x: x[4], reverse=True)

            for idy in range(len(kind_num_bbox_idx)):
                kind_num_bbox_idy = kind_num_bbox_idx[idy]
                if kind_num_bbox_idy[-1] == False:
                    num_bbox_list.append(kind_num_bbox_idy)
                    find_bool = True
                    kind_num_bbox_idy[-1] = find_bool
                    break

    # kind_bbox & num_bbox
    if len(kind_bbox_list):
        kind_bbox_list_x1 = sorted(kind_bbox_list, key=lambda x: x[0])
        kind_bbox_list_y1 = sorted(kind_bbox_list, key=lambda x: x[1])
        kind_bbox_list_x2 = sorted(kind_bbox_list, key=lambda x: x[2], reverse=True)
        kind_bbox_list_y2 = sorted(kind_bbox_list, key=lambda x: x[3], reverse=True)
        kind_bbox.append(kind_bbox_list_x1[0][0])
        kind_bbox.append(kind_bbox_list_y1[0][1])
        kind_bbox.append(kind_bbox_list_x2[0][2])
        kind_bbox.append(kind_bbox_list_y2[0][3])

    if len(num_bbox_list):
        num_bbox_list_x1 = sorted(num_bbox_list, key=lambda x: x[0])
        num_bbox_list_y1 = sorted(num_bbox_list, key=lambda x: x[1])
        num_bbox_list_x2 = sorted(num_bbox_list, key=lambda x: x[2], reverse=True)
        num_bbox_list_y2 = sorted(num_bbox_list, key=lambda x: x[3], reverse=True)
        num_bbox.append(num_bbox_list_x1[0][0])
        num_bbox.append(num_bbox_list_y1[0][1])
        num_bbox.append(num_bbox_list_x2[0][2])
        num_bbox.append(num_bbox_list_y2[0][3])

    # check
    if len(kind_bbox_list) == len(kind_id) and \
        len(num_bbox_list) == len(num_id):

        if len(kind_id) > 0:
            kind_bbox_center_x = (kind_bbox[0] + kind_bbox[2]) / 2.0
            kind_bbox_center_y = (kind_bbox[1] + kind_bbox[3]) / 2.0
            
            if not (kind_bbox_center_x >= num_bbox[0] and \
                    kind_bbox_center_x <= num_bbox[2]) or \
                not (kind_bbox_center_y >= num_bbox[1] and \
                    kind_bbox_center_y <= num_bbox[3]):
                
                kind_num_bool = True
        else:
            # kind: ""
            kind_num_bool = True

    return kind_bbox, num_bbox, kind_num_bool


def decode_xml(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_mask_error_dir)
    create_folder(args.output_mask_img_error_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))
        
        output_mask_path = os.path.join(args.output_mask_dir, img_name.replace(".jpg", ".png"))
        output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name.replace(".jpg", ".png"))
        output_mask_error_path = os.path.join(args.output_mask_error_dir, img_name.replace(".jpg", ".png"))
        output_mask_img_error_path = os.path.join(args.output_mask_img_error_dir, img_name.replace(".jpg", ".png"))

        # img
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=img.dtype)
        mask_img = np.zeros(img.shape, dtype=img.dtype)
    
        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)
        kind_num_bbox_dict = load_kind_num_bbox(xml_path, dataset_dict)

        # kind & num
        ## 获得 kind & num 字段
        kind_num_name = img_name.replace('.jpg', '').split('_')[-1]
        kind_id, num_id = get_kind_num_id(kind_num_name)
        print("{} kind: {}, nmm: {}".format(kind_num_name, kind_id, num_id))

        ## 获得 kind & num 框
        kind_bbox, num_bbox, kind_num_bool = get_kind_num_bbox(kind_id, num_id, kind_num_bbox_dict)

        if len(kind_bbox):
            object_roi_list.append({"classname": "kind", "bndbox":kind_bbox})
        
        if len(num_bbox):
            object_roi_list.append({"classname": "num", "bndbox":num_bbox})

        # 按照一定顺序画图
        # 目的：存在重叠区域
        for class_name_key in dataset_dict.name_2_id_dict.keys():
            for idz in range(len(object_roi_list)):
                classname = object_roi_list[idz]["classname"]
                bndbox = object_roi_list[idz]["bndbox"]

                if classname != class_name_key:
                    continue

                contours = []
                contours.append([bndbox[0], bndbox[1]])
                contours.append([bndbox[2], bndbox[1]])
                contours.append([bndbox[2], bndbox[3]])
                contours.append([bndbox[0], bndbox[3]])
                contours = [np.array(contours).reshape(-1, 1, 2)]

                # kind & num & country & city & car_type
                mask = cv2.drawContours(mask, contours, -1, dataset_dict.name_2_mask_id_dict[classname], cv2.FILLED)
                mask_img = cv2.drawContours(mask_img, contours, -1, dataset_dict.name_2_mask_color_dict[classname], cv2.FILLED)
    
        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)

        if kind_num_bool:
            cv2.imwrite(output_mask_path, mask)
            cv2.imwrite(output_mask_img_path, mask_img)
        else:
            cv2.imwrite(output_mask_error_path, mask)
            cv2.imwrite(output_mask_img_error_path, mask_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop/"
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    # args.seg_dict_name = "dataset_zd_dict"
    # args.output_mask_dir = os.path.join(args.input_dir, 'city_label/mask')
    # args.output_mask_img_dir = os.path.join(args.input_dir, 'city_label/mask_img')
    # args.output_mask_error_dir = os.path.join(args.input_dir, 'city_label/mask_error')
    # args.output_mask_img_error_dir = os.path.join(args.input_dir, 'city_label/mask_img_error')

    args.seg_dict_name = "dataset_zd_dict"
    args.output_mask_dir = os.path.join(args.input_dir, 'city_label_1210/mask')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'city_label_1210/mask_img')
    args.output_mask_error_dir = os.path.join(args.input_dir, 'city_label_1210/mask_error')
    args.output_mask_img_error_dir = os.path.join(args.input_dir, 'city_label_1210/mask_img_error')

    decode_xml(args)

