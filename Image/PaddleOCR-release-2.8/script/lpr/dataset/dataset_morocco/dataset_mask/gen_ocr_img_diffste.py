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

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.lpr.dataset.dataset_morocco.dataset_mask.gen_ocr_img import kind_num_city_mask_2_bbox, get_kind_num_id, get_resize_shape

def gen_img(args, dataset_dict):

    # load diffste data
    diffste_img_list = get_sub_filepaths_suffix(args.diffste_dir, '.jpg')
    diffste_name_list = ["_".join(os.path.basename(diffste_img_path).split('_')[:-7]) + '.jpg' for diffste_img_path in diffste_img_list]
    print("data len: ", len(diffste_img_list))

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "kind": "", "num": "", "crop_img": "", "crop_xml": ", "crop_json": "}]

    # error init 
    error_list = []         # [{"img_path": "", "json_path": "", "crop_img": "", "crop_xml": "", "crop_json": "", "type": "", "value": ""}]

    for _, row in tqdm(args.data_pd.iterrows(), total=len(args.data_pd)):
    
        # info
        img_path = row['img_path']
        json_path = row['json_path']
        roi_img_path = row['roi_img_path']
        roi_mask_path = row['roi_mask_path']
        plate_id = row['id']
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color'] 
        plate_column = row['column'] 
        plate_num = row['num']
        crop_img_path = row['crop_img'] 
        crop_xml_path = row['crop_xml'] 
        # crop_json_path = row['crop_json']
        crop_json_path = crop_xml_path.replace('.xml', '.json').replace('/xml', '/Json')
        split_type = row['split_type']

        crop_img_name = os.path.basename(crop_img_path)
        if crop_img_name not in diffste_name_list:
            continue

        for idx in tqdm(range(len(diffste_img_list))):

            diffste_img_path = diffste_img_list[idx]
            diffste_img_name = "_".join(os.path.basename(diffste_img_path).split('_')[:-7]) + '.jpg'

            roi_img_path = diffste_img_path
            plate_num = os.path.basename(diffste_img_path).split('_')[-1][:-4]
            
            if crop_img_name != diffste_img_name:
                continue
            # print(diffste_img_path)

            # 方案一：加载 xml 文件，得到 kind 和 num 的 bbox
            # # img
            # img = cv2.imread(crop_img_path)
            # # load object_roi_list
            # object_roi_list = dataset_dict.load_object_roi(crop_xml_path)
            # seg_bbox = kind_num_roi_to_bbox(object_roi_list, args.kind_min_size_threh, args.num_min_size_threh)

            # 方案二：从 seg 标签，得到 kind 和 num 的 bbox
            # img
            ori_img = cv2.imread(row['roi_img_path'])
            img = cv2.imread(diffste_img_path)
            # 获取图像高度和宽度
            height, width = img.shape[:2]

            if height == args.to_padding_size[0] and width == args.to_padding_size[1]:

                # 计算裁剪区域的起始位置
                start_h = int((height - args.to_resize_size[1]) / 2)
                end_h = start_h + args.to_resize_size[1]

                # 裁剪图像
                img = img[start_h:end_h, :]
                img = cv2.resize(img, (ori_img.shape[1],ori_img.shape[0]), interpolation=cv2.INTER_CUBIC)

            # mask
            mask = cv2.imread(roi_mask_path)

            # load object_roi_list
            seg_bbox = kind_num_city_mask_2_bbox(mask, dataset_dict, args.kind_min_size_threh, args.num_min_size_threh, args.city_min_size_threh)

            # kind_id & num_id & city_id
            kind_num_name = plate_num
            for replace_name in dataset_dict.replace_name_list:
                kind_num_name = kind_num_name.replace(replace_name, '')
            kind_id, num_id, city_id = get_kind_num_id(kind_num_name)

            # Synchronize
            if not 'kind' in seg_bbox:
                kind_id = ''
                
            if not 'num' in seg_bbox:
                num_id = ''

            if not 'city' in seg_bbox:
                city_id = ''

            # check 
            if not ('kind' in seg_bbox and len(kind_id)) and \
                not ('num' in seg_bbox and len(num_id)) and \
                not ('city' in seg_bbox and len(city_id)):
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "kind_num"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "kind_num", "value": ""})
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

            # city 
            city_img = np.zeros((0, 0))
            if 'city' in seg_bbox and len(city_id):
                city_box = seg_bbox['city'][0]

                x1, x2 = city_box[0], city_box[0]+city_box[2]
                y1, y2 = city_box[1], city_box[1]+city_box[3]
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
                city_img = img[y1:y2, x1:x2]

            try:
                # resize
                kind_shape, num_shape, city_shape = get_resize_shape(kind_img, num_img, city_img)
                if 'kind' in seg_bbox and len(kind_id):
                    kind_img = cv2.resize(kind_img, kind_shape)
                if 'num' in seg_bbox and len(num_id):
                    num_img = cv2.resize(num_img, num_shape)
                if 'city' in seg_bbox and len(city_id):
                    city_img = cv2.resize(city_img, city_shape)

                # concate
                concate_img = np.zeros((0, 0))
                if 'kind' in seg_bbox and len(kind_id) and 'num' in seg_bbox and len(num_id) and 'city' in seg_bbox and len(city_id):
                    concate_img = cv2.hconcat([kind_img, city_img, num_img])     
                elif 'kind' in seg_bbox and len(kind_id) and 'city' in seg_bbox and len(city_id):
                    concate_img = cv2.hconcat([kind_img, city_img]) 
                elif 'city' in seg_bbox and len(city_id) and 'num' in seg_bbox and len(num_id):
                    concate_img = cv2.hconcat([city_img, num_img]) 
                elif 'kind' in seg_bbox and len(kind_id):
                    concate_img = cv2.hconcat([kind_img]) 
                elif 'num' in seg_bbox and len(num_id):
                    concate_img = cv2.hconcat([num_img]) 
            except:
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "crop_json": {}, "type": "resize"'.format(img_path, json_path, crop_img_path, crop_xml_path, crop_json_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "type": "resize", "value": ""})
                continue

            plate_img_name = "{}_{}{}{}.jpg".format(os.path.basename(roi_img_path).replace(".jpg", ""), kind_id, city_id, num_id)
            output_img_path = os.path.join(args.output_img_dir, plate_img_name)
            cv2.imwrite(output_img_path, concate_img)

            csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "kind": kind_id, "num": num_id, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "split_type": split_type})


def gen_ocr_img(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_morocco.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)

    # pd
    args.data_pd = pd.read_csv(args.input_csv_path)

    # gen_img
    gen_img(args, dataset_dict)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="morocco_20250327")
    parser.add_argument('--seg_name', type=str, default="seg_Morocco_202504")
    parser.add_argument('--ocr_name', type=str, default="plate_Morocco_mask_202504")
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    parser.add_argument('--diffste_name', type=str, default="diffste_3000_morocco") 
    parser.add_argument('--diffste_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco_crop_scale_padding/DIFFSTE/") 
    parser.add_argument('--new_style', action='store_true', default=False) 
    args = parser.parse_args()
    args.new_style = True

    args.seg_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.diffste_name)
    args.diffste_dir = os.path.join(args.diffste_dir, args.diffste_name)

    print("gen ocr img.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("output_dir: {}".format(args.output_dir))
    
    args.input_csv_path = os.path.join(args.seg_dir, 'city_label', args.date_name + '.csv')
    args.output_img_dir = os.path.join(args.output_dir, "Images")
    
    args.seg_dict_name = "dataset_morocco_dict_city"

    args.hisi_min_size_threh = 32
    args.kind_min_size_threh = 6
    args.city_min_size_threh = 6
    args.num_min_size_threh = 10
    args.random_expand_ratio = 0.08
    args.to_resize_size = (256, 64)
    args.to_padding_size = (256, 256)

    # 生成 ocr img
    gen_ocr_img(args)