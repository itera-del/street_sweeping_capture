import argparse
import cv2
import importlib
import io
import json
import numpy as np
import os
import random
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_augment.data_augment import *


def find_img_path(img_name, img_list):
    
    for idx in range(len(img_list)):
        img_path = img_list[idx]

        if os.path.basename(img_path) == img_name:
            return img_path
        
    return None


def find_roi(data_json, find_plate_num, dataset_dict):

    for cell in data_json['shapes']:
        # plate points
        if 'points' in cell:
            pts = np.array(cell["points"], np.int32)
            
            if "type" in cell and cell["type"] == "rectangle":
                pts = pts.reshape((-1, 1, 2))
                x1 = np.min((pts[0][0][0], pts[1][0][0]))
                x2 = np.max((pts[0][0][0], pts[1][0][0]))
                y1 = np.min((pts[0][0][1], pts[1][0][1]))
                y2 = np.max((pts[0][0][1], pts[1][0][1]))
            else:
                pts = pts.reshape((-1, 1, 2))
                x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
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
            return True, [x1, y1, x2, y2]
    
    return False, [0, 0, 0, 0]


def draw_mask(dataset_dict, img_crop, object_roi, output_img_path, output_mask_path, output_mask_img_path, output_bbox_img_path):

    mask = np.zeros(img_crop.shape, dtype=img_crop.dtype)
    mask_img = np.zeros(img_crop.shape, dtype=img_crop.dtype)
    bbox_img = img_crop.copy()

    # 按照一定顺序画图
    # 目的：存在重叠区域
    for idx in range(len(dataset_dict.add_mask_order)):
        class_name_key = dataset_dict.add_mask_order[idx]
        for idz in range(len(object_roi)):
            classname = object_roi[idz]["classname"]
            bndbox = object_roi[idz]["bndbox"]

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
            bbox_img = cv2.rectangle(bbox_img, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), color=dataset_dict.name_2_mask_color_dict[classname], thickness=2)

    # mask_img
    mask_img = cv2.addWeighted(src1=img_crop, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)

    cv2.imwrite(output_img_path, img_crop)
    cv2.imwrite(output_mask_path, mask)
    cv2.imwrite(output_mask_img_path, mask_img)
    cv2.imwrite(output_bbox_img_path, bbox_img)


def decode_xml_augument(args):

    # dataset_zd_dict
    # city
    dataset_city_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_city_dict_name) 
    # color
    dataset_color_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_color_dict_name)

    # mkdir
    # city
    create_folder(args.output_city_img_dir)
    create_folder(args.output_city_mask_dir)
    create_folder(args.output_city_mask_img_dir)
    create_folder(args.output_city_bbox_img_dir)
    # color
    create_folder(args.output_color_img_dir)
    create_folder(args.output_color_mask_dir)
    create_folder(args.output_color_mask_img_dir)
    create_folder(args.output_color_bbox_img_dir)

    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # ori img list 
    ori_img_list = get_sub_filepaths_suffix(args.ori_input_dir, ".jpg")
    ori_img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))

        ori_img_path = os.path.join(args.ori_input_dir, ("-").join(img_name.split("-")[:-1]) + '.jpg')
        ori_json_path = ori_img_path.replace(".jpg", ".json")

        try:
            assert os.path.exists(ori_img_path)
            assert os.path.exists(ori_json_path)
            
            # ori_img
            img = cv2.imread(ori_img_path)

            # ori_json 
            with io.open(ori_json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()
        except:
            # 文件集中，可能存在子目录，路径不对
            ori_img_name = ("-").join(img_name.split("-")[:-1]) + '.jpg'
            ori_img_path = find_img_path(ori_img_name, ori_img_list)
            ori_json_path = ori_img_path.replace(".jpg", ".json")

            # ori_img
            img = cv2.imread(ori_img_path)

            # ori_json 
            with io.open(ori_json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()

        # find roi
        plate_num = img_name.replace(".jpg", "").split('_')[-1]
        find_roi_bool, img_roi = find_roi(data_json, plate_num, dataset_city_dict)
        if find_roi_bool == False:
            print(img_path)
            print(ori_img_path)
            print(ori_json_path)
            continue

        # load object_roi_list
        # city
        object_city_roi_list = dataset_city_dict.load_object_roi(xml_path)
        # color
        object_color_roi_list = dataset_color_dict.load_object_roi(xml_path)

        object_roi_list = []
        object_roi_list.extend(object_city_roi_list)
        object_roi_list.extend(object_color_roi_list)

        # aug
        for idy in range(args.aug_times):
            
            # aug_rotate_bbox
            rotate_img, rotate_img_roi, rotate_object_roi = aug_rotate_bbox(img.copy(), img_roi.copy(), object_roi_list.copy(), args.aug_rotate_angle_list)
            # aug_expand_bbox
            expand_img_roi, expand_object_roi = aug_expand_bbox(rotate_img, rotate_img_roi, rotate_object_roi, args.aug_expand_ratio_list)

            rotate_img_crop = rotate_img[expand_img_roi[1]:expand_img_roi[3], expand_img_roi[0]:expand_img_roi[2]]

            # city
            output_city_img_path = os.path.join(args.output_city_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.jpg".format(idy, plate_num)))
            output_city_mask_path = os.path.join(args.output_city_mask_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_city_mask_img_path = os.path.join(args.output_city_mask_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_city_bbox_img_path = os.path.join(args.output_city_bbox_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            draw_mask(dataset_city_dict, rotate_img_crop, expand_object_roi, output_city_img_path, output_city_mask_path, output_city_mask_img_path, output_city_bbox_img_path)
            
            # color
            output_color_img_path = os.path.join(args.output_color_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.jpg".format(idy, plate_num)))
            output_color_mask_path = os.path.join(args.output_color_mask_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_color_mask_img_path = os.path.join(args.output_color_mask_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_color_bbox_img_path = os.path.join(args.output_color_bbox_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            draw_mask(dataset_color_dict, rotate_img_crop, expand_object_roi, output_color_img_path, output_color_mask_path, output_color_mask_img_path, output_color_bbox_img_path)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/") 
    parser.add_argument('--ori_input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_1115_1116/") 
    args = parser.parse_args()

    print("4、gen seg mask augment.")
    print("input_dir: {}".format(args.input_dir))
    print("ori_input_dir: {}".format(args.ori_input_dir))

    ###############################################
    # dataset_zd_dict & dataset_zd_dict_color
    ###############################################
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    args.seg_city_dict_name = "dataset_zd_dict"
    args.output_city_img_dir = os.path.join(args.input_dir, 'city_label/Images_aug')
    args.output_city_mask_dir = os.path.join(args.input_dir, 'city_label/mask_aug')
    args.output_city_mask_img_dir = os.path.join(args.input_dir, 'city_label/mask_img_aug')
    args.output_city_bbox_img_dir = os.path.join(args.input_dir, 'city_label/bbox_img_aug')

    args.seg_color_dict_name = "dataset_zd_dict_color"
    args.output_color_img_dir = os.path.join(args.input_dir, 'color_label/Images_aug')
    args.output_color_mask_dir = os.path.join(args.input_dir, 'color_label/mask_aug')
    args.output_color_mask_img_dir = os.path.join(args.input_dir, 'color_label/mask_img_aug')
    args.output_color_bbox_img_dir = os.path.join(args.input_dir, 'color_label/bbox_img_aug')

    args.aug_times = 5
    args.aug_rotate_angle_list = range(-20, 20)
    args.aug_expand_ratio_list = list(np.arange(0.01, 0.1, 0.01))

    decode_xml_augument(args)