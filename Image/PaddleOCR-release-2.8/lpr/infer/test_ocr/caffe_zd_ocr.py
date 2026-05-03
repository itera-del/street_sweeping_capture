import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from infer.lpr import LPRCaffe, LPRPytorch, ocr_labels_zd


def model_test(args, from_jpg_dir=False):

    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # lpr
    if args.caffe_bool:
        lpr = LPRCaffe(args.lpr_caffe_prototxt, args.lpr_caffe_model_path, input_shape=(256, 64), ocr_labels=ocr_labels_zd, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool)
    elif args.pytorch_bool:
        lpr = LPRPytorch(args.lpr_pth_path, input_shape=(256, 64), ocr_labels=ocr_labels_zd, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool)

    # img list
    img_list = []

    if from_jpg_dir==True:
        img_list = os.listdir(args.input_jpg_path)
        img_list = [img for img in img_list if str(img).endswith('.jpg')]
        img_list = [os.path.join(args.input_jpg_path, img_name) for img_name in img_list]
    else:
        with open(args.img_list) as f:
            for line in f:
                img_list.append(line.strip())   
    
    # results list  
    results_list = []

    for idx in tqdm(range(len(img_list))):
        # init 
        results_dict = {}

        img_name = os.path.basename(img_list[idx])
        img_path = img_list[idx]
        tqdm.write(img_path)

        img = cv2.imread(img_path)        

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # ocr 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr, ocr_score = lpr.run(gray_img)

        # pd 
        results_dict['file'] = img_path
        results_dict['width'] = image_width
        results_dict['height'] = image_height
            
        results_dict['label'] = img_name.replace('.jpg', '').split('_')[-1]

        if args.analysis_label_bool:
            img_name_list = str(img_name).split('_')
            for key_name in args.analysis_label_split_id.keys():
                key_id = args.analysis_label_split_id[key_name]
                key_res = img_name_list[key_id]
                results_dict[key_name] = key_res

        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = np.array(ocr_score).mean()
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )
        # results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )
        
        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], int( results_dict['label'] == results_dict['ocr'] ) )
        tqdm.write(tqdm_write)

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.caffe_bool = False
    args.pytorch_bool = True

    # # zd: ocr_zd_mask_1120
    # args.lpr_caffe_prototxt = ""
    # args.lpr_caffe_model_path = ""
    # args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1120/crnn_best.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1120/"

    # # zd: ocr_zd_mask_20230119
    # args.lpr_caffe_prototxt = ""
    # args.lpr_caffe_model_path = ""
    # args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230119/crnn_best.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230119/"

    # # zd: ocr_zd_mask_20230217
    # args.lpr_caffe_prototxt = ""
    # args.lpr_caffe_model_path = ""
    # args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230217/crnn_best.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230217/"

    # zd: ocr_zd_mask_pad_20230301
    args.lpr_caffe_prototxt = ""
    args.lpr_caffe_model_path = ""
    args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230301/crnn_best.pth"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230301/"

    # # zd: ocr_zd_mask_tcres_20230209
    # args.lpr_caffe_prototxt = ""
    # args.lpr_caffe_model_path = ""
    # args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230209/crnn_best.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230209/"

    # # zd: ocr_zd_mask_tcres_20230217
    # args.lpr_caffe_prototxt = ""
    # args.lpr_caffe_model_path = ""
    # args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230217/crnn_best.pth"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230217/"

    args.lpr_prefix_beam_search_bool = False
    # args.lpr_prefix_beam_search_bool = True

    args.analysis_label_bool = True
    args.analysis_label_split_id = {
                                'country': -5,
                                'city': -4, 
                                'color': -3,
                                'column': -2,
                                }

    # ocr_merge_test
    args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/ImageSetsNoAug/ImageSets/Main/test.txt"
    args.output_csv_path = os.path.join(args.output_dir, 'test/ocr_merge_test_result.csv')
    model_test(args)

    # # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # args.output_csv_path = os.path.join(args.output_dir, 'test/data_synthesis_baozihao_result.csv')
    # model_test(args, from_jpg_dir=True)

    # # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/test"
    # args.output_csv_path = os.path.join(args.output_dir, 'test/data_synthesis_test_result.csv')
    # model_test(args, from_jpg_dir=True)


if __name__ == '__main__':
    main()