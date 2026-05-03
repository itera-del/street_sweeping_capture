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
from Image.recognition2d.lpr.infer.lpr_seg_ocr import LPRSegOcrcaffe

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')


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


def mask_2_bbox(mask):
    
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bkg_id = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            bkg_id = row
    stats_no_bg = np.delete(stats, bkg_id, axis=0)
    stats_no_bg_idx = stats_no_bg[:, 4].argmax()
    bbox = [stats_no_bg[stats_no_bg_idx]]
    
    return bbox


def kind_num_mask_2_bbox(mask, dataset_dict, kind_min_size_threh, num_min_size_threh):
    
    seg_bbox = {}

    for key in dataset_dict.class_seg_label_group_2_id_map.keys():

        if key != "kind" and key != "num":
            continue

        label_list = dataset_dict.class_seg_label_group_2_id_map[key]
        label_mask = np.array((0,0))

        for idy in range(len(label_list)): 
            analysis_label = label_list[idy]
            label_mask = (mask == dataset_dict.id_2_mask_id_dict[analysis_label])[:,:,0].astype(np.uint8)
            label_size = label_mask.sum()
        
        if label_size > 0.0:

            label_bbox = mask_2_bbox(label_mask)[0]
            w = label_bbox[2]
            h = label_bbox[3]

            if key == "kind":
                if w >= kind_min_size_threh and h >= kind_min_size_threh:
                    seg_bbox[key] = [[label_bbox[0], label_bbox[1], w, h]]

            elif key == "num":
                if w >= num_min_size_threh and h >= num_min_size_threh:
                    seg_bbox[key] = [[label_bbox[0], label_bbox[1], w, h]]

    return seg_bbox


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


def gen_img(args, dataset_dict):

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "kind": "", "num": "", "crop_img": "", "crop_xml": ", "crop_json": "}]

    # error init 
    error_list = []         # [{"img_path": "", "json_path": "", "crop_img": "", "crop_xml": "", "crop_json": "", "type": "", "value": ""}]

    for idx, row in tqdm(args.data_pd.iterrows(), total=len(args.data_pd)):
    
        # info
        img_path = row['img_path']
        json_path = row['json_path']
        roi_img_path = row['roi_img_path']
        roi_mask_path = row['roi_mask_path']
        plate_id = row['id']
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_country = row['country'] 
        plate_city = row['city'] 
        plate_color = row['color'] 
        plate_column = row['column'] 
        plate_num = row['num'] 
        crop_img_path = row['crop_img'] 
        crop_xml_path = row['crop_xml'] 
        # crop_json_path = row['crop_json']
        crop_json_path = crop_xml_path.replace('.xml', '.json').replace('/xml', '/Json')
        split_type = row['split_type']

        # 方案：加载分割模型得到图像
        # img
        img = cv2.imread(roi_img_path)

        seg_bbox, seg_info, ocr, ocr_score, ocr_ignore, concate_img = args.lpr_seg_ocr.run(img)
    
        # kind_id & num_id
        kind_num_name = plate_num
        for replace_name in dataset_dict.replace_name_list:
            kind_num_name = kind_num_name.replace(replace_name, '')
        kind_id, num_id = get_kind_num_id(kind_num_name)
        kind_id_res, num_id_res = get_kind_num_id(ocr)

        if kind_id == "" and kind_id_res != "" and \
             num_id == num_id_res:
            kind_id = "^"
            
            plate_img_name = "{}_{}#{}.jpg".format(os.path.basename(roi_img_path).replace(".jpg", ""), kind_id, num_id)
            output_img_path = os.path.join(args.output_img_dir, plate_img_name)
            cv2.imwrite(output_img_path, concate_img)

            csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "country": plate_country, "city": plate_city, "color": plate_color, "column": plate_column, "kind": kind_id, "num": num_id, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path, "split_type": split_type})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_pd = pd.DataFrame(error_list)
    out_error_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd.to_csv(out_error_csv_path, index=False, encoding="utf_8_sig")


def gen_ocr_img(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)     

    # pd
    args.data_pd = pd.read_csv(args.input_csv_path)
    
    # model
    args.lpr_seg_ocr = LPRSegOcrcaffe(args.seg_caffe_prototxt, args.seg_caffe_model_path, args.ocr_caffe_prototxt, args.ocr_caffe_model_path, gpu_bool=args.gpu_bool, dict_path=args.dict_path)

    # gen_img
    gen_img(args, dataset_dict)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="uae_20241105")
    parser.add_argument('--seg_name', type=str, default="seg_zd_202502")
    parser.add_argument('--ocr_name', type=str, default="plate_zd_mask_202502")
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.seg_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.date_name + "_seg_res")

    print("gen ocr img.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("output_dir: {}".format(args.output_dir))

    args.input_csv_path = os.path.join(args.seg_dir, 'city_label', args.date_name + '.csv')
    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_csv_path = os.path.join(args.output_dir, '{}.csv'.format(args.date_name))
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data")
    
    args.seg_dict_name = "dataset_zd_dict_city_2022"

    # seg model
    args.gpu_bool = True
    args.seg_caffe_prototxt = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_aug_20250320/zd_seg_city_class_color_aug_20250320.prototxt"
    args.seg_caffe_model_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_aug_20250320/zd_seg_city_class_color_aug_20250320.caffemodel"
    args.ocr_caffe_prototxt = "/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250320_Aug/inference_300/onnx/paddle_ocr_zd_mask_aug_20250320.prototxt"
    args.ocr_caffe_model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_zd_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20250320_Aug/inference_300/onnx/paddle_ocr_zd_mask_aug_20250320.caffemodel"
    args.dict_path = "/yuanhuan/model/image/lpr/paddle_dict/plate_zd_mask_20250320/zd_dict.txt"

    # 生成 ocr img
    gen_ocr_img(args)
