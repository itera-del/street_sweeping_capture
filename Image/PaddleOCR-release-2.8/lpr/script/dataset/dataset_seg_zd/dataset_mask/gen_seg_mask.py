import argparse
import cv2
import importlib
import numpy as np
import os
import sys
from tqdm import tqdm


# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def get_kind_num_id(kind_num_name):
    # init
    kind_id, num_id  = '', ''

    img_name_list = kind_num_name.split('#')
    if len(img_name_list) == 1:
        num_id = img_name_list[0]
    elif len(img_name_list) == 2:
        if len(img_name_list[0]) < len(img_name_list[1]):
            kind_id = img_name_list[0]
            num_id = img_name_list[1]
        else:
            kind_id = img_name_list[1]
            num_id = img_name_list[0]

    return kind_id, num_id


def decode_xml(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_bbox_img_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))
        
        output_mask_path = os.path.join(args.output_mask_dir, img_name.replace(".jpg", ".png"))
        output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name.replace(".jpg", ".png"))
        output_bbox_img_path = os.path.join(args.output_bbox_img_dir, img_name.replace(".jpg", ".png"))

        # img
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=img.dtype)
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        bbox_img = img.copy()
        
        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)

        # 按照一定顺序画图
        # 目的：存在重叠区域
        for idy in range(len(dataset_dict.add_mask_order)):
            class_name_key = dataset_dict.add_mask_order[idy]
            for idz in range(len(object_roi_list)):
                classname = object_roi_list[idz]["classname"]
                bndbox = object_roi_list[idz]["bndbox"]

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
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)

        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)
        cv2.imwrite(output_bbox_img_path, bbox_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_1115_1116/") 
    args = parser.parse_args()

    print("3、gen seg mask.")
    print("input_dir: {}".format(args.input_dir))

    ###############################################
    # dataset_zd_dict
    ###############################################
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")
    
    args.seg_dict_name = "dataset_zd_dict"
    args.output_mask_dir = os.path.join(args.input_dir, 'city_label/mask')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'city_label/mask_img')
    args.output_bbox_img_dir = os.path.join(args.input_dir, 'city_label/bbox_img')

    decode_xml(args)

    ###############################################
    # dataset_zd_dict_color
    ###############################################
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    args.seg_dict_name = "dataset_zd_dict_color"
    args.output_mask_dir = os.path.join(args.input_dir, 'color_label/mask')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'color_label/mask_img')
    args.output_bbox_img_dir = os.path.join(args.input_dir, 'color_label/bbox_img')

    decode_xml(args)