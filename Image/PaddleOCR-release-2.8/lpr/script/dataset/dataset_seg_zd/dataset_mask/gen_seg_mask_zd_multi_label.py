import argparse
import cv2
import importlib
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def decode_xml(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 
    
    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(args.input_xml_dir, img_name.replace(".jpg", ".xml"))
        
        output_mask_path = os.path.join(args.output_mask_dir, img_name.replace(".jpg", ".pkl"))
        output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name.replace(".jpg", ".png"))

        # img
        img = cv2.imread(img_path)
        
        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)

        # 生成 mask_sigmoid & mask_sigmoid_img，multi_label
        mask_sigmoid = np.zeros((img.shape[0], img.shape[1], len(dataset_dict.class_seg_sigmoid_label)), dtype=img.dtype)
        mask_sigmoid_img = np.zeros(img.shape, dtype=img.dtype)

        for class_name_key in dataset_dict.name_2_sigmoid_id_dict.keys():
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

                # country & country_f & city & city_f & car_type & car_type_f & color & kind & num
                mask_roi = mask_sigmoid[:, :, dataset_dict.name_2_sigmoid_id_dict[classname]]
                mask_roi = np.ascontiguousarray(mask_roi)
                mask_sigmoid[:, :, dataset_dict.name_2_sigmoid_id_dict[classname]] = cv2.drawContours(mask_roi, contours, -1, 1, cv2.FILLED)
                
                mask_img_roi = np.zeros(img.shape, dtype=img.dtype)
                mask_img_roi = cv2.drawContours(mask_img_roi, contours, -1, dataset_dict.name_2_sigmoid_mask_color_dict[classname], cv2.FILLED)
                mask_sigmoid_img += mask_img_roi

        # mask_sigmoid_img
        mask_sigmoid_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_sigmoid_img, beta=0.3, gamma=0.)

        f = open(output_mask_path, 'wb')
        pickle.dump(mask_sigmoid, f)
        f.close()

        cv2.imwrite(output_mask_img_path, mask_sigmoid_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/data_crop/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/data_crop_0804_0809/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/data_crop_0810_0811/"
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_xml_dir = os.path.join(args.input_dir, "xml")

    args.seg_dict_name = "dataset_zd_dict_multi_label"
    args.output_mask_dir = os.path.join(args.input_dir, 'multi_label/mask_sigmoid')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'multi_label/mask_sigmoid_img')

    decode_xml(args)