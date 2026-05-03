import argparse
import cv2
import importlib
import io
import json
import numpy as np
import numpy.random as npr
import os
import pickle
import random
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_augment.data_augment import *


def find_roi(data_json, find_plate_num, dataset_dict):

    for cell in data_json['shapes']:
        # plate points
        if 'points' in cell:
            pts = np.array(cell["points"], np.int32)
            
            if cell["type"] == "rectangle":
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
        elif 'attributes' in cell:
            plate_num = cell["attributes"][0]["value"]

        plate_num = plate_num.upper()
        for replace_name in dataset_dict.replace_name_list:
            plate_num = plate_num.replace(replace_name, '')

        if plate_num == find_plate_num:
            return True, [x1, y1, x2, y2]
    
    return False, [0, 0, 0, 0]


def aug_rotate_bbox(img, img_roi, object_roi, aug_rotate_angle_list):

    # img_roi
    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    points = np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))

    # object_roi
    for idx in range(len(object_roi)):
        bndbox = object_roi[idx]["bndbox"]
        x1 = bndbox[0] + img_roi[0]
        y1 = bndbox[1] + img_roi[1]
        x2 = bndbox[2] + img_roi[0]
        y2 = bndbox[3] + img_roi[1]
        points = np.concatenate((points, np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))), axis=1)

    # aug_rotate_angle
    aug_rotate_angle = random.sample(aug_rotate_angle_list, 1)[0]
    
    # rotate_img_with_points
    rotate_img, rotate_points = rotate_img_with_points(img, points, aug_rotate_angle)
    
    # rotate_img_roi
    x1 = min(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y1 = min(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    x2 = max(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y2 = max(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    rotate_img_roi = [x1, y1, x2, y2]
    
    # rotate_object_roi
    rotate_object_roi = []
    for idx in range(len(object_roi)):
        classname = object_roi[idx]["classname"]

        x1 = min(rotate_points[0][4 + idx*4], rotate_points[0][4 + idx*4 + 1], rotate_points[0][4 + idx*4 + 2], rotate_points[0][4 + idx*4 + 3]) - rotate_img_roi[0]
        y1 = min(rotate_points[1][4 + idx*4], rotate_points[1][4 + idx*4 + 1], rotate_points[1][4 + idx*4 + 2], rotate_points[1][4 + idx*4 + 3]) - rotate_img_roi[1]
        x2 = max(rotate_points[0][4 + idx*4], rotate_points[0][4 + idx*4 + 1], rotate_points[0][4 + idx*4 + 2], rotate_points[0][4 + idx*4 + 3]) - rotate_img_roi[0]
        y2 = max(rotate_points[1][4 + idx*4], rotate_points[1][4 + idx*4 + 1], rotate_points[1][4 + idx*4 + 2], rotate_points[1][4 + idx*4 + 3]) - rotate_img_roi[1]
        bndbox = [x1, y1, x2, y2]

        rotate_object_roi.append({"classname": classname, "bndbox":bndbox})

    return rotate_img, rotate_img_roi, rotate_object_roi


def aug_expand_bbox(img, img_roi, object_roi, aug_expand_ratio_list):

    # aug_expand_ratio
    aug_expand_ratio = random.sample(aug_expand_ratio_list, 1)[0]

    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    h = y2 - y1
    w = x2 - x1

    # expand_img_roi
    x1 = max(0, x1 + npr.randint(-aug_expand_ratio * w, 1))
    x2 = min(img.shape[1], x2 + npr.randint(-1, aug_expand_ratio * w))
    y1 = max(0, y1 + npr.randint(-aug_expand_ratio * h, 1))
    y2 = min(img.shape[0], y2 + npr.randint(-1, aug_expand_ratio * h))
    expand_img_roi = [x1, y1, x2, y2]

    # expand_object_roi
    expand_object_roi = []
    for idx in range(len(object_roi)):
        classname = object_roi[idx]["classname"]
        bndbox = object_roi[idx]["bndbox"]

        x1 = bndbox[0] + img_roi[0] - expand_img_roi[0]
        y1 = bndbox[1] + img_roi[1] - expand_img_roi[1]
        x2 = bndbox[2] + img_roi[0] - expand_img_roi[0]
        y2 = bndbox[3] + img_roi[1] - expand_img_roi[1]
        bndbox = [x1, y1, x2, y2]
        expand_object_roi.append({"classname": classname, "bndbox":bndbox})

    return expand_img_roi, expand_object_roi


def decode_xml_augument(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)

    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))

        ori_img_path = os.path.join(args.ori_input_dir, ("-").join(img_name.split("-")[:-1]) + '.jpg')
        ori_json_path = ori_img_path.replace(".jpg", ".json")

        # ori_img
        img = cv2.imread(ori_img_path)

        # ori_json 
        with io.open(ori_json_path, "r", encoding="UTF-8") as f:
            data_json = json.load(f, encoding='utf-8')
            f.close()

        # find roi
        plate_num = img_name.replace(".jpg", "").split('_')[-1]
        find_roi_bool, img_roi = find_roi(data_json, plate_num, dataset_dict)
        assert find_roi_bool

        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)

        # aug
        for idy in range(args.aug_times):
            
            output_img_path = os.path.join(args.output_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.jpg".format(idy, plate_num)))
            output_mask_path = os.path.join(args.output_mask_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.pkl".format(idy, plate_num)))
            output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))

            # aug_rotate_bbox
            rotate_img, rotate_img_roi, rotate_object_roi = aug_rotate_bbox(img.copy(), img_roi.copy(), object_roi_list.copy(), args.aug_rotate_angle_list)
            # aug_expand_bbox
            expand_img_roi, expand_object_roi = aug_expand_bbox(rotate_img, rotate_img_roi, rotate_object_roi, args.aug_expand_ratio_list)

            # crop
            rotate_img_crop = rotate_img[expand_img_roi[1]:expand_img_roi[3], expand_img_roi[0]:expand_img_roi[2]]

            mask_sigmoid = np.zeros((rotate_img_crop.shape[0], rotate_img_crop.shape[1], len(dataset_dict.class_seg_sigmoid_label)), dtype=img.dtype)
            mask_sigmoid_img = np.zeros(rotate_img_crop.shape, dtype=img.dtype)

            # 生成 mask_sigmoid & mask_sigmoid_img，multi_label
            for class_name_key in dataset_dict.name_2_sigmoid_id_dict.keys():
                for idz in range(len(expand_object_roi)):
                    classname = expand_object_roi[idz]["classname"]
                    bndbox = expand_object_roi[idz]["bndbox"]

                    if classname != class_name_key:
                        continue

                    contours = []
                    contours.append([bndbox[0], bndbox[1]])
                    contours.append([bndbox[2], bndbox[1]])
                    contours.append([bndbox[2], bndbox[3]])
                    contours.append([bndbox[0], bndbox[3]])
                    contours = [np.array(contours).reshape(-1, 1, 2)]

                    # country & country_f & city & city_f & car_type & car_type_f & color & kind & num
                    mask_roi = mask_sigmoid[:, :, dataset_dict.name_2_sigmoid_id_dict[classname]]
                    mask_roi = np.ascontiguousarray(mask_roi)
                    mask_sigmoid[:, :, dataset_dict.name_2_sigmoid_id_dict[classname]] = cv2.drawContours(mask_roi, contours, -1, 1, cv2.FILLED)
                    
                    mask_img_roi = np.zeros(rotate_img_crop.shape, dtype=rotate_img_crop.dtype)
                    mask_img_roi = cv2.drawContours(mask_img_roi, contours, -1, dataset_dict.name_2_sigmoid_mask_color_dict[classname], cv2.FILLED)
                    mask_sigmoid_img += mask_img_roi

            # mask_sigmoid_img
            mask_sigmoid_img = cv2.addWeighted(src1=rotate_img_crop, alpha=0.8, src2=mask_sigmoid_img, beta=0.3, gamma=0.)

            cv2.imwrite(output_img_path, rotate_img_crop)

            f = open(output_mask_path, 'wb')
            pickle.dump(mask_sigmoid, f)
            f.close()

            cv2.imwrite(output_mask_img_path, mask_sigmoid_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.ori_input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0804_0809/"
    args.ori_input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0810_0811/"

    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/data_crop_0804_0809/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/data_crop_0810_0811/"

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    args.seg_dict_name = "dataset_zd_dict_multi_label"
    args.output_img_dir = os.path.join(args.input_dir, 'multi_label/Images_aug')
    args.output_mask_dir = os.path.join(args.input_dir, 'multi_label/mask_sigmoid_aug')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'multi_label/mask_sigmoid_img_aug')
    
    args.aug_times = 5
    args.aug_rotate_angle_list = range(-20, 20)
    args.aug_expand_ratio_list = list(np.arange(0.01, 0.1, 0.01))
    
    decode_xml_augument(args)