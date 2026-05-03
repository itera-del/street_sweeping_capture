import argparse
import cv2
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def gen_ocr_img(args):

    # mkdir
    create_folder(args.output_img_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]
    
    # img list
    img_list = get_sub_filepaths_suffix(args.input_dir, 'jpg')
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        # info
        img_path = img_path
        json_path = ''
        plate_id = 0
        plate_name = img_name.split('.jpg')[0]
        plate_roi = []
        plate_color = ''
        # plate_column = ''
        plate_column = 'single'
        plate_num = str(img_name).replace('.jpg', '').split('_')[-1]
        # print(plate_num)

        # plate_img_name = str(plate_name) + '.jpg'
        plate_img_name = str(plate_name).replace(plate_num, "{}_{}".format(plate_column, plate_num)) + '.jpg'
        output_img_path = os.path.join(args.output_img_dir, plate_img_name)

        if args.crop_bool:
            # img
            img = cv2.imread(img_path)

            # 获取图像高度和宽度
            height, width = img.shape[:2]

            if height == args.to_padding_size[0] and width == args.to_padding_size[1]:

                # 计算裁剪区域的起始位置
                start_h = int((height - args.to_resize_size[1]) / 2)
                end_h = start_h + args.to_resize_size[1]

                # 裁剪图像
                cropped_image = img[start_h:end_h, :]

                # 保存
                # print("crop_bool: {}, cropped_image.shape: {}/{}".format(args.crop_bool, cropped_image.shape[0], cropped_image.shape[1]))
                cv2.imwrite(output_img_path, cropped_image)
            else:
                # print("crop_bool: {}, img.shape: {}/{}".format(args.crop_bool, img.shape[0], img.shape[1]))
                shutil.copy(img_path, output_img_path)

        else:
            # print("crop_bool: {}".format(args.crop_bool))
            shutil.copy(img_path, output_img_path)
        csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="diffste_huanwei_test_100w")
    parser.add_argument('--ocr_name', type=str, default="plate_huanwei_20240923")
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_HUANWEI/original/Argentina/DIFFSTE/")
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_HUANWEI/training_ocr/")
    parser.add_argument('--crop_bool', type=bool, default=True)
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.date_name)

    print("gen ocr img.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_csv_path = os.path.join(args.output_dir, '{}.csv'.format(args.date_name))
    args.to_resize_size = (256, 64)     # diffste_huanwei_test_40w diffste_huanwei_20240910_20240911_1018_60w diffste_huanwei_20240912_323_20w
    # args.to_resize_size = (256, 128)  # diffste_huanwei_20250820_20250925_3w
    args.to_padding_size = (256, 256)

    # 生成 ocr img
    gen_ocr_img(args)
