import argparse
import cv2
import importlib
import numpy as np
import os
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
from infer.lpr_seg import LPRSegCaffe, LPRSegPytorch, LPRSeg2HeadPytorch, LPRSegColorClassPytorch


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
        lpr_seg = LPRSegCaffe(args.seg_caffe_prototxt, args.seg_caffe_model_path, args.seg_dict_name)
    elif args.pytorch_bool:
        lpr_seg = LPRSegPytorch(args.seg_pth_path, args.seg_dict_name)

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
        preds_mask, seg_mask, seg_bbox, seg_info = lpr_seg.run(img)

        # mask_img
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask == mask_id] = dataset_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        
        # mask
        mask = np.zeros((64, 128, 3), dtype=preds_mask.dtype)
        for mask_id in dataset_dict.id_2_mask_id_dict.keys():
            mask[preds_mask == mask_id] = dataset_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)

        # bbox_img
        for key in seg_bbox.keys():
            bbox = seg_bbox[key][0]
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_dict.name_2_mask_color_dict[key], thickness=2)
        output_bbox_img_path = os.path.join(args.output_bbox_img, img_name)
        cv2.imwrite(output_bbox_img_path, img)

        # result
        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        for label_idx in range(len(dataset_dict.class_seg_label_group)):
            results_dict[dataset_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_dict.class_seg_label_group[label_idx] + '_res'] = 'none'
    
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
            for label_idx in range(len(dataset_dict.class_seg_label_group)):
                if classname in dataset_dict.class_seg_label_group_2_name_map[dataset_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_dict.class_seg_label_group[label_idx]] = classname
        
        ## 获得模型预测结果字段
        for label_idx in range(len(dataset_dict.class_seg_label_group)):
            results_dict[dataset_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_dict.class_seg_label_group[label_idx]]
            
        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


def model_test_color_class(args, from_jpg_dir=False):
    
    # dataset_zd_dict 
    # city
    dataset_city_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_city_dict_name) 
    # color
    dataset_color_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_color_dict_name)

    # mkdir 
    create_folder(args.output_dir)
    create_folder(args.output_mask_city)
    create_folder(args.output_mask_color)
    create_folder(args.output_mask_img_city)
    create_folder(args.output_mask_img_color)
    create_folder(args.output_bbox_img)

    # init 
    if args.caffe_bool:
        raise NotImplementedError
    elif args.pytorch_bool:
        lpr_seg = LPRSegColorClassPytorch(args.seg_pth_path, args.seg_city_dict_name, args.seg_color_dict_name)

    # img list
    img_list = []

    if from_jpg_dir==True:
        img_list = os.listdir(args.input_jpg_path)
        img_list = [os.path.join(args.input_jpg_path, img_name) for img_name in img_list]
    else:
        with open(args.input_test_file_path) as f:
            for line in f:
                img_list.append(line.strip().split(".jpg ")[0] + '.jpg')   

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
        # city
        object_city_roi_list = dataset_city_dict.load_object_roi(xml_path)
        # color
        object_color_roi_list = dataset_color_dict.load_object_roi(xml_path)
        
        # run
        preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox, seg_info = lpr_seg.run(img)

        # mask_img
        # city
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_city_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_1 == mask_id] = dataset_city_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        # color
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_color_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_2 == mask_id] = dataset_color_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=1.0, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        
        # mask
        # city
        mask = np.zeros((64, 128, 3), dtype=preds_mask_1.dtype)
        for mask_id in dataset_city_dict.id_2_mask_id_dict.keys():
            mask[preds_mask_1 == mask_id] = dataset_city_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)
        # color
        mask = np.zeros((64, 128, 3), dtype=preds_mask_2.dtype)
        for mask_id in dataset_color_dict.id_2_mask_id_dict.keys():
            if preds_mask_2.max() == mask_id:
                mask[:, :] = dataset_color_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)

        # bbox_img
        for key in seg_bbox.keys():
            bbox = seg_bbox[key][0]

            # city
            if key in dataset_city_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_city_dict.name_2_mask_color_dict[key], thickness=2)
            # color
            elif key in dataset_color_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_color_dict.name_2_mask_color_dict[key], thickness=2)
        output_bbox_img_path = os.path.join(args.output_bbox_img, img_name)
        cv2.imwrite(output_bbox_img_path, img)

        # result
        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = 'none'
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = 'none'

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
        for idz in range(len(object_city_roi_list)):
            classname = object_city_roi_list[idz]["classname"]
            bndbox = object_city_roi_list[idz]["bndbox"]

            # city
            for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
                if classname in dataset_city_dict.class_seg_label_group_2_name_map[dataset_city_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = classname

        for idz in range(len(object_color_roi_list)):
            classname = object_color_roi_list[idz]["classname"]
            bndbox = object_color_roi_list[idz]["bndbox"]

            # color
            for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
                if classname in dataset_color_dict.class_seg_label_group_2_name_map[dataset_color_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = classname

        ## 获得模型预测结果字段
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_city_dict.class_seg_label_group[label_idx]]
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_color_dict.class_seg_label_group[label_idx]]
            
        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


def model_test_2_head(args, from_jpg_dir=False):
    
    # dataset_zd_dict 
    # city
    dataset_city_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_city_dict_name) 
    # color
    dataset_color_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_color_dict_name)

    # mkdir 
    create_folder(args.output_dir)
    create_folder(args.output_mask_city)
    create_folder(args.output_mask_color)
    create_folder(args.output_mask_img_city)
    create_folder(args.output_mask_img_color)
    create_folder(args.output_bbox_img)

    # init 
    if args.caffe_bool:
        raise NotImplementedError
    elif args.pytorch_bool:
        lpr_seg = LPRSeg2HeadPytorch(args.seg_pth_path, args.seg_city_dict_name, args.seg_color_dict_name)

    # img list
    img_list = []

    if from_jpg_dir==True:
        img_list = os.listdir(args.input_jpg_path)
        img_list = [os.path.join(args.input_jpg_path, img_name) for img_name in img_list]
    else:
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
        # city
        object_city_roi_list = dataset_city_dict.load_object_roi(xml_path)
        # color
        object_color_roi_list = dataset_color_dict.load_object_roi(xml_path)
        
        # run
        preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox, seg_info = lpr_seg.run(img)

        # mask_img
        # city
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_city_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_1 == mask_id] = dataset_city_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        # color
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_color_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_2 == mask_id] = dataset_color_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=1.0, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        
        # mask
        # city
        mask = np.zeros((64, 128, 3), dtype=preds_mask_1.dtype)
        for mask_id in dataset_city_dict.id_2_mask_id_dict.keys():
            mask[preds_mask_1 == mask_id] = dataset_city_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)
        # color
        mask = np.zeros((64, 128, 3), dtype=preds_mask_2.dtype)
        for mask_id in dataset_color_dict.id_2_mask_id_dict.keys():
            mask[preds_mask_2 == mask_id] = dataset_color_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)

        # bbox_img
        for key in seg_bbox.keys():
            bbox = seg_bbox[key][0]

            # city
            if key in dataset_city_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_city_dict.name_2_mask_color_dict[key], thickness=2)
            # color
            elif key in dataset_color_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_color_dict.name_2_mask_color_dict[key], thickness=2)
        output_bbox_img_path = os.path.join(args.output_bbox_img, img_name)
        cv2.imwrite(output_bbox_img_path, img)

        # result
        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = 'none'
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = 'none'

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
        for idz in range(len(object_city_roi_list)):
            classname = object_city_roi_list[idz]["classname"]
            bndbox = object_city_roi_list[idz]["bndbox"]

            # city
            for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
                if classname in dataset_city_dict.class_seg_label_group_2_name_map[dataset_city_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = classname

        for idz in range(len(object_color_roi_list)):
            classname = object_color_roi_list[idz]["classname"]
            bndbox = object_color_roi_list[idz]["bndbox"]

            # color
            for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
                if classname in dataset_color_dict.class_seg_label_group_2_name_map[dataset_color_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = classname

        ## 获得模型预测结果字段
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_city_dict.class_seg_label_group[label_idx]]
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_color_dict.class_seg_label_group[label_idx]]
            
        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # ###############################################
    # # dataset_zd_dict
    # ###############################################
    # args.caffe_bool = False
    # args.pytorch_bool = True

    # # # zd: seg_city_cartype_kind_num_zd_0826
    # # args.seg_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/LaneNetNova_class_15.prototxt"
    # # args.seg_caffe_model_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/LaneNetNova_seg_city_cartype_kind_num_zd_0826.caffemodel"
    # # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/LaneNetNova.pth"
    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/"
    
    # # zd: seg_city_cartype_kind_num_zd_1019
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019/LaneNetNova.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019/"
    
    # args.seg_dict_name = "dataset_zd_dict"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/"
    # args.dataset_name = "merge_test_0804_0809_0810_0811"
    # args.mode = "test"
    # args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))

    # args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.output_mask = os.path.join(args.output_dir, '{}/{}/mask'.format(args.mode, args.dataset_name))
    # args.output_mask_img = os.path.join(args.output_dir, '{}/{}/mask_img'.format(args.mode, args.dataset_name))
    # args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    # model_test(args)

    # ###############################################
    # # dataset_zd_dict_color
    # ###############################################

    # args.caffe_bool = False
    # args.pytorch_bool = True

    # # zd: seg_color_zd_1101
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1101/LaneNetNova.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1101/"
    
    # args.seg_dict_name = "dataset_zd_dict_color"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/"
    # args.dataset_name = "merge_test_0804_0809_0810_0811"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))
    
    # args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.output_mask = os.path.join(args.output_dir, '{}/{}/mask'.format(args.mode, args.dataset_name))
    # args.output_mask_img = os.path.join(args.output_dir, '{}/{}/mask_img'.format(args.mode, args.dataset_name))
    # args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    # model_test(args)


    # ###############################################
    # # dataset_zd_dict & dataset_zd_dict_color
    # ###############################################

    # args.caffe_bool = False
    # args.pytorch_bool = True

    # # # zd: seg_city_color_zd_1103
    # # args.seg_caffe_prototxt = ""
    # # args.seg_caffe_model_path = "" 
    # # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1103/LaneNetNova2Head.pth"
    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1103/"

    # # zd: seg_city_color_zd_1104
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1104/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1104/"
    
    # # from dataset
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/"
    # args.dataset_name = "merge_test_0804_0811"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))
    
    # args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.output_mask_city = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    # args.output_mask_color = os.path.join(args.output_dir, '{}/{}/mask_color'.format(args.mode, args.dataset_name))
    # args.output_mask_img_city = os.path.join(args.output_dir, '{}/{}/mask_img_city'.format(args.mode, args.dataset_name))
    # args.output_mask_img_color = os.path.join(args.output_dir, '{}/{}/mask_img_color'.format(args.mode, args.dataset_name))
    # args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    # model_test_2_head(args)

    # # from_jpg_dir
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"
    # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/"
    # args.dataset_name = "jpg文件"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_jpg_path = os.path.join(args.input_dir, args.dataset_name)
    
    # args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.output_mask_city = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    # args.output_mask_color = os.path.join(args.output_dir, '{}/{}/mask_color'.format(args.mode, args.dataset_name))
    # args.output_mask_img_city = os.path.join(args.output_dir, '{}/{}/mask_img_city'.format(args.mode, args.dataset_name))
    # args.output_mask_img_color = os.path.join(args.output_dir, '{}/{}/mask_img_color'.format(args.mode, args.dataset_name))
    # args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    # model_test_2_head(args, from_jpg_dir=True)


    ###############################################
    # dataset_zd_dict & dataset_zd_dict_color
    # CityColorClassSeg
    ###############################################

    args.caffe_bool = False
    args.pytorch_bool = True

    # # zd: seg_city_color_class_zd_1107
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1107/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1107/"
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"

    # # zd: seg_city_color_class_no_grad_zd_1108
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_no_grad_zd_1108/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_no_grad_zd_1108/"
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"

    # # zd: seg_city_color_class_zd_1117
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1117/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1117/"
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"

    # # zd: seg_city_color_class_zd_1210
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1210/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1210/"
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"

    # # zd: seg_city_color_class_zd_20230119
    # args.seg_caffe_prototxt = ""
    # args.seg_caffe_model_path = "" 
    # args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230119/LaneNetNova2Head.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230119/"
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"

    # zd: seg_city_color_class_zd_20230217
    args.seg_caffe_prototxt = ""
    args.seg_caffe_model_path = "" 
    args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230217/LaneNetNova2Head.pth"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230217/"
    args.seg_city_dict_name = "dataset_zd_dict"
    args.seg_color_dict_name = "dataset_zd_dict_color"

    # from dataset
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd_202301/"
    args.dataset_name = "ImageSetsCityColorLabelNoAug"
    args.mode = "test"
    # args.mode = "trainval"
    args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))
    
    args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    args.output_mask_city = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    args.output_mask_color = os.path.join(args.output_dir, '{}/{}/mask_color'.format(args.mode, args.dataset_name))
    args.output_mask_img_city = os.path.join(args.output_dir, '{}/{}/mask_img_city'.format(args.mode, args.dataset_name))
    args.output_mask_img_color = os.path.join(args.output_dir, '{}/{}/mask_img_color'.format(args.mode, args.dataset_name))
    args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    model_test_color_class(args)

    # # from_jpg_dir
    # args.seg_city_dict_name = "dataset_zd_dict"
    # args.seg_color_dict_name = "dataset_zd_dict_color"
    # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/"
    # args.dataset_name = "jpg文件"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_jpg_path = os.path.join(args.input_dir, args.dataset_name, 'color')
    
    # args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.output_mask_city = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    # args.output_mask_color = os.path.join(args.output_dir, '{}/{}/mask_color'.format(args.mode, args.dataset_name))
    # args.output_mask_img_city = os.path.join(args.output_dir, '{}/{}/mask_img_city'.format(args.mode, args.dataset_name))
    # args.output_mask_img_color = os.path.join(args.output_dir, '{}/{}/mask_img_color'.format(args.mode, args.dataset_name))
    # args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    # model_test_color_class(args, from_jpg_dir=True)