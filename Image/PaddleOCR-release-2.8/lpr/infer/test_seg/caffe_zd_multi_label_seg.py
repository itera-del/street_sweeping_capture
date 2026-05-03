import argparse
import cv2
import importlib
import numpy as np
import os
import pickle
import pandas as pd
import sys 
from tqdm import tqdm

# caffe_root = '/home/huanyuan/code/caffe/'
# sys.path.insert(0, caffe_root + 'python')  
# import caffe

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_mask.gen_seg_mask import get_kind_num_id
from infer.lpr_multi_label_seg import LPRSegColorPytorch

    
def model_test(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir 
    create_folder(args.output_dir)
    create_folder(args.output_mask)
    create_folder(args.output_mask_img)
    create_folder(args.output_bbox_img)

    # init 
    if args.caffe_bool:
        # lpr_seg = LPRSegCaffe(args.city_seg_caffe_prototxt, args.city_seg_caffe_model_path)
        raise NotImplementedError
    elif args.pytorch_bool:
        lpr_seg = LPRSegColorPytorch(args.city_seg_pth_path, args.seg_dict_name)

    # img list
    img_list = []
    with open(args.input_test_file_path) as f:
        for line in f:
            img_list.append(line.strip().split(" ")[0])   

    # results list 
    results_list = []

    for idx in tqdm(range(len(img_list))):
        # init 
        results_dict = {}

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), "xml", img_name.replace(".jpg", ".xml"))
        tqdm.write(img_path)

        img = cv2.imread(img_path) 

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]
        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(xml_path)
        
        # run
        pred_mask, seg_mask, seg_bbox, seg_info = lpr_seg.run(img)

        # mask_img
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for label_idx in dataset_dict.id_2_sigmoid_mask_color_dict.keys():
            mask_img[seg_mask[:,:,label_idx] == 1] = dataset_dict.id_2_sigmoid_mask_color_dict[label_idx]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        
        # mask
        output_mask_path = os.path.join(args.output_mask, img_name.replace(".jpg", ".pkl"))
        f = open(output_mask_path, 'wb')
        pickle.dump(pred_mask, f)
        f.close()

        # bbox_img
        for key in seg_bbox.keys():
            bbox = seg_bbox[key][0]
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_dict.sigmoid_name_id_2_sigmoid_mask_color_dict[key], thickness=2)
            # img = cv2.putText(img, "{}".format( key ), (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 1, dataset_dict.sigmoid_name_id_2_sigmoid_mask_color_dict[key], 2)
        output_bbox_img_path = os.path.join(args.output_bbox_img, img_name)
        cv2.imwrite(output_bbox_img_path, img)

        # result
        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        for label_idx in range(len(dataset_dict.class_seg_sigmoid_label)):
            results_dict[dataset_dict.class_seg_sigmoid_label[label_idx]] = 'none'
            results_dict[dataset_dict.class_seg_sigmoid_label[label_idx] + '_res'] = 'none'
    
        ## 获得 kind & num 字段
        if 'kind' in results_dict.keys() and 'num' in results_dict.keys():
            kind_num_name = img_name.replace('.jpg', '').split('_')[-1]
            kind_id, num_id = get_kind_num_id(kind_num_name)
            print("{} kind: {}, nmm: {}".format(kind_num_name, kind_id, num_id))

            if kind_id != '':
                results_dict['kind'] = 'kind'

            if num_id != '':
                results_dict['num'] = 'num'

        ## 获得 country & city & car_type & color 字段
        for idz in range(len(object_roi_list)):
            classname = object_roi_list[idz]["classname"]
            bndbox = object_roi_list[idz]["bndbox"]

            for label_idx in range(len(dataset_dict.class_seg_sigmoid_label)):
                if classname in dataset_dict.class_seg_label_sigmoid_2_name_map[dataset_dict.class_seg_sigmoid_label[label_idx]]:
                    results_dict[dataset_dict.class_seg_sigmoid_label[label_idx]] = dataset_dict.class_seg_sigmoid_label[label_idx]

        ## 获得模型预测结果字段
        for label_idx in range(len(dataset_dict.class_seg_sigmoid_label)):
            results_dict[dataset_dict.class_seg_sigmoid_label[label_idx] + '_res'] = seg_info[dataset_dict.class_seg_sigmoid_label[label_idx]]

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.caffe_bool = False
    args.pytorch_bool = True

    # zd: seg_city_cartype_color_kind_num_zd_1101
    args.city_seg_caffe_prototxt = ""
    args.city_seg_caffe_model_path = "" 
    args.city_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_mutil_seg_zd_1107/LaneNetNova.pth"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_mutil_seg_zd_1107/"
    
    args.seg_dict_name = "dataset_zd_dict_multi_label"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_mutil_seg/"
    args.dataset_name = "sigmoid_merge_test"
    args.mode = "test"
    # args.mode = "trainval"
    args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))

    
    args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    args.output_mask = os.path.join(args.output_dir, '{}/{}/mask'.format(args.mode, args.dataset_name))
    args.output_mask_img = os.path.join(args.output_dir, '{}/{}/mask_img'.format(args.mode, args.dataset_name))
    args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    model_test(args)
