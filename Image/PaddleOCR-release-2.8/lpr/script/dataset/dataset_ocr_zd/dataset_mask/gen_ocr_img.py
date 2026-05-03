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


def load_interval_img(args):
    
    interval_img = np.zeros((0, 0))
    ori_interval_img = cv2.imread(args.interval_img_path)
    with io.open(args.interval_json_path, "r", encoding="UTF-8") as f:
        data = json.load(f, encoding='utf-8')
        f.close()
    pts = np.array(data['shapes'][0]["points"], np.int32).reshape((-1, 1, 2))
    x1 = np.min(pts[:, 0, 0])
    x2 = np.max(pts[:, 0, 0])
    y1 = np.min(pts[:, 0, 1])
    y2 = np.max(pts[:, 0, 1])
    interval_img = ori_interval_img[y1:y2, x1:x2]

    return interval_img


def load_end_img(args):

    end_img = np.zeros((0, 0))
    ori_end_img = cv2.imread(args.end_img_path)
    with io.open(args.end_json_path, "r", encoding="UTF-8") as f:
        data = json.load(f, encoding='utf-8')
        f.close()
    pts = np.array(data['shapes'][0]["points"], np.int32).reshape((-1, 1, 2))
    x1 = np.min(pts[:, 0, 0])
    x2 = np.max(pts[:, 0, 0])
    y1 = np.min(pts[:, 0, 1])
    y2 = np.max(pts[:, 0, 1])
    end_img = ori_end_img[y1:y2, x1:x2]

    return end_img


def get_resize_shape(kind_img, num_img, interval_img):
# def get_resize_shape(kind_img, num_img, interval_img, end_img):

    # init
    aligned_height = max(kind_img.shape[0], num_img.shape[0])
    
    # rate
    kind_rate = aligned_height / (kind_img.shape[0] + 1e-5)
    num_rate = aligned_height / (num_img.shape[0] + 1e-5)
    interval_rate = aligned_height / (interval_img.shape[0] + 1e-5)
    # end_rate = aligned_height / (end_img.shape[0] + 1e-5)

    kind_shape = []
    kind_shape.append(int(kind_img.shape[1] * kind_rate))
    kind_shape.append(aligned_height)
    
    num_shape = []
    num_shape.append(int(num_img.shape[1] * num_rate))
    num_shape.append(aligned_height)
    
    interval_shape = []
    interval_shape.append(int(interval_img.shape[1] * interval_rate))
    interval_shape.append(aligned_height)

    # end_shape = []
    # end_shape.append(int(end_img.shape[1] * end_rate))
    # end_shape.append(aligned_height)

    return kind_shape, num_shape, interval_shape
    # return kind_shape, num_shape, interval_shape, end_shape


def kind_num_roi_to_bbox(roi_list, kind_min_size_threh, num_min_size_threh):

    # init
    roi_dict = {}

    for idx in range(len(roi_list)):
        classname = roi_list[idx]["classname"]
        bndbox = roi_list[idx]["bndbox"]
        w = bndbox[2] - bndbox[0]
        h = bndbox[3] - bndbox[1]

        if classname == "kind":
            if w >= kind_min_size_threh and h >= kind_min_size_threh:
                roi_dict[classname] = [[bndbox[0], bndbox[1], w, h]]

        elif classname == "num":
            if w >= num_min_size_threh and h >= num_min_size_threh:
                roi_dict[classname] = [[bndbox[0], bndbox[1], w, h]]
        
        else:
            continue
    
    return roi_dict
    

def decode_xml(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # img list
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
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
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))
        tqdm.write(img_path)

        # img
        img = cv2.imread(img_path) 
        
        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)
        seg_bbox = kind_num_roi_to_bbox(object_roi_list, args.kind_min_size_threh, args.num_min_size_threh)

        # kind_id & num_id
        kind_num_name = img_name.replace('.jpg', '').split('_')[-1]
        for replace_name in dataset_dict.replace_name_list:
            kind_num_name = kind_num_name.replace(replace_name, '')
        kind_id, num_id = get_kind_num_id(kind_num_name)
        # print("{} kind: {}, num: {}".format(kind_num_name, kind_id, num_id))

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

            x1 = max(0, x1 + npr.randint(-args.random_expand_ratio * 2 *  w, 0))
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_202301/data_crop_0804_0809/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/data_crop_0804_0809/")
    args = parser.parse_args()

    print("3、gen ocr img.")
    print("input_dir: {}".format(args.input_dir))

    args.seg_dict_name = "dataset_zd_dict"
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")

    args.interval_img_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.jpg"
    args.interval_json_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.json"
    # args.end_img_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/0000000000000000-220829-154340-154610-00000F000180_59.61_66.00_00080-00_none_none_none_Single_7641.jpg"
    # args.end_json_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/0000000000000000-220829-154340-154610-00000F000180_59.61_66.00_00080-00_none_none_none_Single_7641.json"

    args.hisi_min_size_threh = 32
    args.kind_min_size_threh = 6
    args.num_min_size_threh = 10
    args.random_expand_ratio = 0.08

    decode_xml(args)