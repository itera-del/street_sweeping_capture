import argparse
import cv2
import importlib
import io
import json
import numpy as np
import numpy.random as npr
import os
import pandas as pd
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_mask.gen_seg_mask import get_kind_num_id
from script.dataset.dataset_ocr_zd.dataset_mask.gen_ocr_img_augment import kind_num_mask_2_bbox


def gen_kind_ocr_img(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name)

    for dataset_idx in tqdm(range(len(args.input_dir_list))):

        dataset_dir = args.input_dir_list[dataset_idx]

        # img list
        img_list = get_sub_filepaths_suffix(os.path.join(dataset_dir, args.input_img_dir), ".jpg")
        img_list.sort()

        # run
        for idx in tqdm(range(len(img_list))):

            img_path = img_list[idx]
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(dataset_dir, args.input_mask_dir, img_name.replace(".jpg", ".png"))
            tqdm.write(img_path)

            # img
            img = cv2.imread(img_path) 

            # mask
            mask = cv2.imread(mask_path)

            # load object_roi_list
            seg_bbox = kind_num_mask_2_bbox(mask, dataset_dict, args.kind_min_size_threh, args.num_min_size_threh)

            # kind_id & num_id
            kind_num_name = img_name.replace('.jpg', '').split('_')[-1]
            for replace_name in dataset_dict.replace_name_list:
                kind_num_name = kind_num_name.replace(replace_name, '')
            kind_id, num_id = get_kind_num_id(kind_num_name)
            print("{} kind: {}, num: {}".format(kind_num_name, kind_id, num_id))

            # Synchronize
            if not 'kind' in seg_bbox:
                kind_id = ''

            # kind
            kind_img = np.zeros((0, 0))
            if 'kind' in seg_bbox and len(kind_id):
                kind_box = seg_bbox['kind'][0]

                x1, x2 = kind_box[0], kind_box[0]+kind_box[2]
                y1, y2 = kind_box[1], kind_box[1]+kind_box[3]
                h = y2 - y1
                w = x2 - x1

                # 适配海思开发板，最小 roi 大小
                if w < args.hisi_min_size_threh:
                    x1 = int(((x1 + x2) / 2) - (args.hisi_min_size_threh / 2) + 0.5)
                    x2 = x1 + args.hisi_min_size_threh
                    w = args.hisi_min_size_threh

                if h < args.hisi_min_size_threh:
                    y1 = int(((y1 + y2) / 2) - (args.hisi_min_size_threh / 2) + 0.5)
                    y2 = y1 + args.hisi_min_size_threh
                    h = args.hisi_min_size_threh

                x1 = max(0, x1 + npr.randint(-args.random_expand_ratio * 2 *  w, 0))
                x2 = min(img.shape[1], x2 + npr.randint(0, args.random_expand_ratio * 2 * w))
                y1 = max(0, y1 + npr.randint(-args.random_expand_ratio * 2 * h, 0))
                y2 = min(img.shape[0], y2 + npr.randint(0, args.random_expand_ratio * 2 * h))
                kind_img = img[y1:y2, x1:x2]
            
            # save
            if 'kind' in seg_bbox and len(kind_id):
                if kind_id in args.find_kind_list:
                    output_img_name = "{}_{}.jpg".format('_'.join(img_name.split('_')[:-1]), kind_id)
                    output_img_path = os.path.join(args.output_dir, args.output_kind_format.format(kind_id), output_img_name)
                    create_folder(os.path.dirname(output_img_path))
                    cv2.imwrite(output_img_path, kind_img)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir_list = [
        # "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop/",
        "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0804_0809/",
        "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0810_0811/",
        "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0828_0831/",
        "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0901_0903/",
        "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0904_0905/",
    ]
    
    args.input_img_dir = "city_label/Images_aug"
    args.input_mask_dir = "city_label/mask_aug"

    args.find_kind_list = ['X', 'Y', 'Z']

    args.seg_dict_name = "dataset_zd_dict"
    args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_synthesis/"
    args.output_kind_format = "data_kind_aug/{}"

    args.hisi_min_size_threh = 32
    args.kind_min_size_threh = 6
    args.num_min_size_threh = 10
    args.random_expand_ratio = 0.08

    gen_kind_ocr_img(args)