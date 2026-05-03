import argparse
import cv2
import io
import numpy as np
import json
import os
import pandas as pd
import re
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_huanwei.dataset_dict.dataset_huanwei_dict_normal import *


def dataset_csv(args):
    # mkdir
    create_folder(args.output_csv_dir)
    create_folder(args.output_error_data_dir)

    # img list
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # init
    csv_list = []  # [{"img_path": "", "json_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]
    error_list = []  # [{"img_path": "", "json_path": "", "type": "", "value": ""}]

    for idx in tqdm(range(len(img_list))):

        plate_id = 0
        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]
        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)

        if args.bool_check_img:
            # img
            try:
                img = cv2.imread(img_path)
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "img", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "img"'.format(img_path, json_path))
                raise Exception()
                continue

        # json
        try:
            with io.open(json_path, "r") as f:
                data_json = json.load(f)
                f.close()
        except:
            error_list.append({"img_path": img_path, "json_path": json_path, "type": "json", "value": ""})
            print('"img_path": {}, "json_path": {},"type": "json"'.format(img_path, json_path))
            # raise Exception()
            continue

        for cell in data_json['shapes']:

            # plate num
            plate_num = str(json_load_object_plate_num(cell))
            if 'none' in plate_num.lower() or plate_num.lower() == '':
                continue

            # plate roi
            try:
                x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)
                plate_roi = str("{},{},{},{}".format(x1, x2, y1, y2))
                if args.bool_check_img:
                    plate_img = img[y1:y2, x1:x2]
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_roi", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "plate_roi"'.format(img_path, json_path))
                raise Exception()
                continue

            # plate status
            plate_status = json_load_object_plate_status(cell)
            if plate_status != status_name_list[0]:
                continue

            # ======================== 核心修改部分 ========================
            # plate num check
            
            # 1. 校验是否只由 0-9, A-Z, # 组成
            # bool_valid_chars = bool(re.match(r'^[0-9A-Z#]+$', plate_num))
            
            # 2. 校验是否包含至少一个数字
            bool_has_digit = bool(re.search(r'[0-9]', plate_num))
            
            # 3. 校验是否包含至少一个大写字母
            bool_has_letter = bool(re.search(r'[A-Z]', plate_num))


            # 必须同时满足上述 3 个条件才算作合法车牌
            bool_plate_valid = bool_has_digit and bool_has_letter

            if not bool_plate_valid:
                error_list.append(
                    {"img_path": img_path, "json_path": json_path, "type": "plate_num", "value": plate_num})
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path,
                                                                                                plate_num))
                # raise Exception()
                continue
            # ==============================================================

            # test
            # if plate_num in args.error_plate_num_list or "I" in plate_num or "O" in plate_num or "Q" in plate_num:
            if plate_num in args.error_plate_num_list:
                error_list.append(
                    {"img_path": img_path, "json_path": json_path, "type": "error_plate_num", "value": plate_num})
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path,
                                                                                                plate_num))
                # raise Exception()
                continue

            plate_name = args.data_format.format(img_name, plate_id, plate_num)
            plate_id += 1

            csv_list.append(
                {"img_path": img_path, "json_path": json_path, "id": plate_id, "name": plate_name, "roi": plate_roi,
                 "num": str(plate_num)})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(error_data_csv_path, index=False, encoding="utf_8_sig")


def write_crop_data(args):
    if not args.bool_write_crop_data:
        return

    # mkdir
    create_folder(args.output_crop_data_img_dir)

    # pd
    data_pd = pd.read_csv(args.output_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        plate_name = row['name']
        plate_roi = row['roi']

        # img
        img = cv2.imread(img_path)

        # plate_img
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])
        plate_img = img[y1:y2, x1:x2]

        # 图像旋转90度
        if (y2 - y1) > (x2 - x1) * 1.25:
            plate_img = cv2.rotate(plate_img, cv2.ROTATE_90_CLOCKWISE)

        try:
            cv2.imwrite(os.path.join(args.output_crop_data_img_dir, plate_name + ".jpg"), plate_img)
        except:
            continue


def write_error_data(args):
    if not args.bool_write_error_data:
        return

    # init
    platform_json_writer = PlatformJsonWriter()

    # mkdir
    create_folder(args.output_error_data_img_dir)
    create_folder(args.output_error_data_json_dir)

    # pd
    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')

    try:
        data_pd = pd.read_csv(error_data_csv_path)
    except:
        return

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # init
        rect_list = []

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        type = row['type']
        img_name = os.path.basename(img_path)
        json_name = os.path.basename(json_path)

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_img_dir, type, img_name)
            create_folder(os.path.dirname(to_img_path))
            shutil.copy(img_path, to_img_path)
        except:
            continue

        # json
        try:
            out_json_path = os.path.join(args.output_error_data_json_dir, type, json_name)
            create_folder(os.path.dirname(out_json_path))
            if type == "json":
                # write empty platform json
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_height, img_width = img.shape[0], img.shape[1]
                platform_json_writer.write_json(
                    img_width,
                    img_height,
                    img_name,
                    out_json_path,
                    frame_num=idx,
                    rect_list=[]
                )
            else:
                shutil.copy(json_path, out_json_path)
        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Argentina_20250910_无编号有垃圾桶")
    parser.add_argument('--input_dir', type=str,
                        default="/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina/")
    parser.add_argument('--output_csv_dir', type=str,
                        default="/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_csv/")
    parser.add_argument('--output_crop_data_dir', type=str,
                        default="/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_crop/")
    parser.add_argument('--output_error_data_dir', type=str,
                        default="/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_error_data/")
    parser.add_argument('--bool_write_crop_data', action='store_true', default=False)
    parser.add_argument('--bool_write_error_data', action='store_true', default=False)
    parser.add_argument('--bool_check_img', action='store_true', default=False)
    parser.add_argument('--img_folder', type=str, default="JPEGImages1")
    parser.add_argument('--json_folder', type=str, default="Jsons")
    args = parser.parse_args()

    args.data_format = "{}-{:0>2d}_{}"  # name-id_车牌号
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.output_csv_path = os.path.join(args.output_csv_dir, args.date_name + '.csv')
    args.output_crop_data_dir = os.path.join(args.output_crop_data_dir, args.date_name)
    args.output_error_data_dir = os.path.join(args.output_error_data_dir, args.date_name)
    args.error_plate_num_list = []

    print("dataset csv.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.img_dir))
    print("output_csv_path: {}".format(args.output_csv_path))

    # 生成 dataset csv
    dataset_csv(args)

    # 保存 crop data
    args.output_crop_data_img_dir = os.path.join(args.output_crop_data_dir, "Images")
    write_crop_data(args)

    # 保存 error data
    args.output_error_data_img_dir = os.path.join(args.output_error_data_dir, "Images")
    args.output_error_data_json_dir = os.path.join(args.output_error_data_dir, "Jsons")
    write_error_data(args)