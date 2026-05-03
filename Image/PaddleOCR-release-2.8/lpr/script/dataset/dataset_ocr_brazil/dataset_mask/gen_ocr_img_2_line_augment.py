import argparse
import cv2
import io
import json
import numpy as np
import numpy.random as npr
import os
import pandas as pd
import random
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_augment.data_augment import *
from script.dataset.dataset_ocr_brazil.dataset_mask.gen_ocr_img_augment import find_roi, aug_rotate_bbox, aug_expand_bbox


def dataset_augument(args):
    
    # mkdir
    create_folder(args.output_img_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        ori_img_path = os.path.join(args.input_dir, ("-").join(img_name.split("-")[:-1]) + '.jpg')
        ori_json_path = ori_img_path.replace(".jpg", ".json")

        # print(img_name)
        # print(ori_img_path)

        # ori_img
        img = cv2.imread(ori_img_path)

        # ori_json 
        with io.open(ori_json_path, "r", encoding="UTF-8") as f:
            data_json = json.load(f, encoding='utf-8')
            f.close()

        # find roi
        plate_num = img_name.split('-')[-1].split("_")[-2].split('.jpg')[0]
        plate_kind_id = img_name.split('-')[-1].split("_")[-1].split('.jpg')[0]
        find_roi_bool, img_roi = find_roi(data_json, plate_num)
        assert find_roi_bool

        # plate num
        if len(plate_num) != 7:
            print("[ERROR]: plate_num:{}, len: {}/7. ".format(plate_num, len(plate_num)))
            continue
        plate_kind = plate_num[:3]
        plata_id = plate_num[3:]

        # aug
        for idy in range(args.aug_times):

            # aug_rotate_bbox
            rotate_img, rotate_img_roi = aug_rotate_bbox(img.copy(), img_roi.copy(), args.aug_rotate_angle_list)
            # aug_expand_bbox
            expand_img_roi = aug_expand_bbox(rotate_img, rotate_img_roi, args.aug_expand_ratio_list)

            # crop
            rotate_img_crop = rotate_img[expand_img_roi[1]:expand_img_roi[3], expand_img_roi[0]:expand_img_roi[2]]
            img_h = rotate_img_crop.shape[0]
            img_w = rotate_img_crop.shape[1]
            
            # kind & id
            if plate_kind_id == plate_kind:
                output_img_kind_path = os.path.join(args.output_img_dir, img_name.replace("_{}_{}.jpg".format(plate_num, plate_kind), "_aug_{}_{}_{}.jpg".format(idy, plate_num, plate_kind)))
            
                img_kind = rotate_img_crop[: int(img_h * args.kind_id_height_thres_list[0]), :, :]
                cv2.imwrite(output_img_kind_path, img_kind)
            
            elif plate_kind_id == plata_id:
                output_img_id_path = os.path.join(args.output_img_dir, img_name.replace("_{}_{}.jpg".format(plate_num, plata_id), "_aug_{}_{}_{}.jpg".format(idy, plate_num, plata_id)))

                img_id = rotate_img_crop[int(img_h * args.kind_id_height_thres_list[1]) :, :, :]
                cv2.imwrite(output_img_id_path, img_id)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/2022_0914/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_brazil_2_line/data_crop_2022_0914/")
    args = parser.parse_args()

    print("2、gen ocr img 2 line augment.")
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.output_dir, "Images")
    args.output_img_dir = os.path.join(args.output_dir, "Images_aug")

    args.aug_times = 5
    args.aug_rotate_angle_list = range(-20, 20)
    args.aug_expand_ratio_list = list(np.arange(0.02, 0.1, 0.02))
    args.kind_id_height_thres_list = [0.65, 0.45]

    dataset_augument(args)