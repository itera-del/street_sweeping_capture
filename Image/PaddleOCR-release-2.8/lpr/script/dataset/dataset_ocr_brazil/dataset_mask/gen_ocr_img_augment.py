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

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_augment.data_augment import *


def find_roi(data_json, find_plate_num):
    
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
            if len(cell["attributes"]):
                plate_num = cell["attributes"][0]["value"]

        plate_num = plate_num.upper()

        if plate_num == find_plate_num:
            return True, [x1, y1, x2, y2]
    
    return False, [0, 0, 0, 0]


def dataset_augument(args):

    # mkdir
    create_folder(args.output_img_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    # ori img list 
    ori_img_list = get_sub_filepaths_suffix(args.input_dir, ".jpg")
    ori_img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        ori_img_path = os.path.join(args.input_dir, ("-").join(img_name.split("-")[:-1]) + '.jpg')
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
        find_roi_bool, img_roi = find_roi(data_json, plate_num)
        if find_roi_bool == False:
            print(img_path)
            print(ori_img_path)
            print(ori_json_path)
            continue

        # plate num
        if len(plate_num) != 7:
            print("[ERROR]: plate_num:{}, len: {}/7. ".format(plate_num, len(plate_num)))
            continue
        
        # aug
        for idy in range(args.aug_times):

            output_img_path = os.path.join(args.output_img_dir, img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.jpg".format(idy, plate_num)))

            # aug_rotate_roi
            rotate_img, rotate_img_roi = aug_rotate_roi(img.copy(), img_roi.copy(), args.aug_rotate_angle_list)
            # aug_expand_roi
            expand_img_roi = aug_expand_roi(rotate_img, rotate_img_roi, args.aug_expand_ratio_list)

            # crop
            rotate_img_crop = rotate_img[expand_img_roi[1]:expand_img_roi[3], expand_img_roi[0]:expand_img_roi[2]]

            cv2.imwrite(output_img_path, rotate_img_crop)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/2022_0914/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_brazil/data_crop_2022_0914/") 
    args = parser.parse_args()

    print("2、gen ocr img augment.")
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.output_dir, "Images")
    args.output_img_dir = os.path.join(args.output_dir, "Images_aug")

    args.aug_times = 5
    args.aug_rotate_angle_list = range(-20, 20)
    args.aug_expand_ratio_list = list(np.arange(0.02, 0.1, 0.02))

    dataset_augument(args)