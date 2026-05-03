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
from script.dataset.dataset_ocr_zd.dataset_mask.gen_ocr_img import load_interval_img, load_end_img, get_resize_shape
from infer.lpr_seg import LPRSegCaffe, LPRSegPytorch


def pre_img(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # init  
    if args.caffe_bool:
        lpr_seg = LPRSegCaffe(args.seg_caffe_prototxt, args.seg_caffe_model_path, args.seg_dict_name)
    elif args.pytorch_bool:
        lpr_seg = LPRSegPytorch(args.seg_pth_path, args.seg_dict_name)

    # img list
    img_list = get_sub_filepaths_suffix(args.input_img_dir, suffix='.jpg')
    img_list.sort()

    # error init 
    error_list = []

    # interval
    interval_img = load_interval_img(args)

    # # end
    # end_img = load_end_img(args)

    # run
    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        tqdm.write(img_path)

        img = cv2.imread(img_path) 

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # run
        preds_mask, seg_mask, seg_bbox, seg_info = lpr_seg.run(img)
        
        # kind_id & num_id
        kind_num_name = img_name.replace('.jpg', '').split('_')[-1]
        for replace_name in dataset_dict.replace_name_list:
            kind_num_name = kind_num_name.replace(replace_name, '')
        kind_id, num_id = get_kind_num_id(kind_num_name)
        print("{} kind: {}, num: {}".format(kind_num_name, kind_id, num_id))

        # Synchronize
        if not 'kind' in seg_bbox:
            kind_id = ''
            
        if not 'num' in seg_bbox:
            num_id = ''

        # check 
        if not ('kind' in seg_bbox and len(kind_id)) and \
            not ('num' in seg_bbox and len(num_id)):
            error_list.append({"img_path": img_path, "type": "kind_num"})
            continue
        
        # check 
        if len(num_id):
            error_num_bool = np.array([num not in dataset_dict.num_labels for num in num_id]).sum()
            if error_num_bool:
                error_list.append({"img_path": img_path, "type": "num"})
                continue

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

            x1 = max(0, x1 + npr.randint(-args.random_expand_ratio * 2 * w, 0))
            x2 = min(img.shape[1], x2 + npr.randint(0, args.random_expand_ratio * 2 * w))
            y1 = max(0, y1 + npr.randint(-args.random_expand_ratio * 2 * h, 0))
            y2 = min(img.shape[0], y2 + npr.randint(0, args.random_expand_ratio * 2 * h))
            kind_img = img[y1:y2, x1:x2]
        
        # num 
        num_img = np.zeros((0, 0))
        if 'num' in seg_bbox and len(num_id):
            num_box = seg_bbox['num'][0]

            x1, x2 = num_box[0], num_box[0]+num_box[2]
            y1, y2 = num_box[1], num_box[1]+num_box[3]
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

            x1 = max(0, x1 + npr.randint(-args.random_expand_ratio * 2 * w, 0))
            x2 = min(img.shape[1], x2 + npr.randint(0, args.random_expand_ratio * 2 * w))
            y1 = max(0, y1 + npr.randint(-args.random_expand_ratio * h, 0))
            y2 = min(img.shape[0], y2 + npr.randint(0, args.random_expand_ratio * h))
            num_img = img[y1:y2, x1:x2]

        try:
            # resize
            kind_shape, num_shape, interval_shape = get_resize_shape(kind_img, num_img, interval_img)
            # kind_shape, num_shape, interval_shape, end_shape = get_resize_shape(kind_img, num_img, interval_img, end_img)
            if 'kind' in seg_bbox and len(kind_id):
                kind_img = cv2.resize(kind_img, kind_shape)
            if 'num' in seg_bbox and len(num_id):
                num_img = cv2.resize(num_img, num_shape)
            interval_img_copy = cv2.resize(interval_img, interval_shape)
            # end_img_copy = cv2.resize(end_img, end_shape)

            # concate
            concate_img = np.zeros((0, 0))
            if 'kind' in seg_bbox and len(kind_id) and 'num' in seg_bbox and len(num_id):
                concate_img = cv2.hconcat([kind_img, interval_img_copy, num_img])     
                # concate_img = cv2.hconcat([kind_img, interval_img_copy, num_img, end_img_copy])     
            elif 'kind' in seg_bbox and len(kind_id):
                concate_img = cv2.hconcat([kind_img, interval_img_copy]) 
                # concate_img = cv2.hconcat([kind_img, interval_img_copy, end_img_copy]) 
            elif 'num' in seg_bbox and len(num_id):
                concate_img = cv2.hconcat([interval_img_copy, num_img]) 
                # concate_img = cv2.hconcat([interval_img_copy, num_img, end_img_copy]) 
        except:
            error_list.append({"img_path": img_path, "type": "resize"})
            continue

        output_img_name = "{}_{}#{}.jpg".format('_'.join(img_name.split('_')[:-1]), kind_id, num_id)
        # output_img_name = "{}_{}#{}@.jpg".format('_'.join(img_name.split('_')[:-1]), kind_id, num_id)
        output_img_path = os.path.join(args.output_img_dir, output_img_name)
        cv2.imwrite(output_img_path, concate_img)

    # out csv
    out_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/data_crop/")
    args = parser.parse_args()

    print("add、dataset pre img ocr.")
    print("input_dir: {}".format(args.input_dir))

    args.caffe_bool = False
    args.pytorch_bool = True

    # zd: seg_city_cartype_kind_num_zd_1220
    args.seg_caffe_prototxt = ""
    args.seg_caffe_model_path = ""
    args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1220/LaneNetNova.pth"
    args.seg_dict_name = "dataset_zd_dict"

    args.input_img_dir = os.path.join(args.input_dir, "Images")

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    args.interval_img_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.jpg"
    args.interval_json_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.json"
    # args.end_img_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/0000000000000000-220829-154340-154610-00000F000180_59.61_66.00_00080-00_none_none_none_Single_7641.jpg"
    # args.end_json_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/0000000000000000-220829-154340-154610-00000F000180_59.61_66.00_00080-00_none_none_none_Single_7641.json"

    args.hisi_min_size_threh = 32
    args.random_expand_ratio = 0.08

    pre_img(args)


if __name__ == '__main__':
    main()