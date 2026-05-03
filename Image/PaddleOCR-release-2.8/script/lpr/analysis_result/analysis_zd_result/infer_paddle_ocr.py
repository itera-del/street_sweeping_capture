import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.paddle.infer.lpr import LPRPaddle, LPROnnx, LPRCaffe


def model_test(args, from_jpg_dir=False, bool_onnx=False, bool_caffe=False):

    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # lpr
    if bool_onnx:
        lpr = LPROnnx(args.model_path, args.config_path, padding_bool=False)
    elif bool_caffe:
        lpr = LPRCaffe(args.model_path, args.prototxt_path, args.dict_path, img_shape=args.img_shape, padding_bool=False)
    else:
        lpr = LPRPaddle(args.config_path, args.model_path)

    # img list
    img_list = []

    if from_jpg_dir==True:
        img_list = os.listdir(args.input_jpg_path)
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

        if bool_onnx or bool_caffe:
            img = cv2.imread(img_path)
        else:
            with open(img_path, 'rb') as f:
                img = f.read()

        ocr, ocr_score = lpr.run(img)

        # pd 
        results_dict['file'] = img_path
        results_dict['label'] = img_name.replace('.jpg', '').split('_')[-1].replace(' - 副本', '').replace('-', '').replace('^', '')
        
        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = np.array(ocr_score).mean()
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )
        # results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )

        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], results_dict['res'] )
        tqdm.write(tqdm_write)

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # ###############################
    # # paddle
    # ###############################
    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/latest"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/inference/"

    # # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_eu_202403A/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/trainval.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_paddle/data_eu_trainval_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_eu_202403A/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_paddle/data_eu_test_result.csv')
    # model_test(args)


    # ###############################
    # # onnx
    # ###############################

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_aug/best_accuracy/"

    # # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_eu_202403A/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/data_eu_test_result.csv')
    # model_test(args, bool_onnx=True)

    # # # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/data_synthesis_baozihao_result.csv')
    # model_test(args, from_jpg_dir=True, bool_onnx=True)


    ###############################
    # caffe
    ###############################

    #######
    # cnn
    #######
    # diffste_248_111600
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/cn_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/best_accuracy/"
    
    # # original_248_52
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/cn_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/best_accuracy/"

    # # diffste_1_7M(diffste_2141_yellow/diffste_2189_green/diffste_3859_blue)
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M/inference_iter_epoch_175/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M/inference_iter_epoch_175/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M/inference_iter_epoch_175/cn_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M/inference_iter_epoch_175/"

    # # ocr_test
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/train.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_train_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_test_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/ImageSetsOcrLabelNoAugWoXianggangaomenDoubleyellow/ImageSets/Main/train.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_all_no_xianggangaomen_doubleyellow_train_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/ImageSetsOcrLabelNoAugWoXianggangaomenDoubleyellow/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_all_no_xianggangaomen_doubleyellow_test_result.csv')
    # model_test(args, bool_caffe=True)

    # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/cn/DIFFSTE/original_248_52/train"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_original_248_52_train_result.csv')
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/cn/DIFFSTE/original_248_52/val"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_original_248_52_val_result.csv')
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/Images/"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_result.csv')
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/Images/"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_result_no_char.csv')
    # # model_test(args, from_jpg_dir=True, bool_caffe=True)

    # #######
    # # brazil
    # #######
    # # data + diffste(200,000)
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/brazil_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/"

    # # ocr_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_brazil_202309/ImageSetsBrazilnewstyleNoAug_single_line/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_test_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_brazil_202309/ImageSetsBrazilnewstyleNoAug_single_line/ImageSets/Main/trainval.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_trainval_result.csv')
    # model_test(args, bool_caffe=True)

    # # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/brazil_02210_202301/Images"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_original_result.csv')
    # # model_test(args, from_jpg_dir=True, bool_caffe=True)

    #######
    # us
    #######
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202310_w_diffste_w_aug/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202310_w_diffste_w_aug/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202310_w_diffste_w_aug/inference/us_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202310_w_diffste_w_aug/inference/"
    # args.img_shape = (1, 128, 256)

    args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/inference/caffe/model.caffemodel"
    args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/inference/caffe/model.prototxt"
    args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/inference/us_dict.txt"
    args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_us_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202406_ALL_US/inference/"
    args.img_shape = (1, 128, 256)

    # ocr_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202310/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_test_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202310/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/trainval.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_trainval_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202406/Alabama_2022/ImageSets/Main/all.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_Alabama_all_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202406/Alabama_2022/ImageSets/Main/all_threshold.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_Alabama_all_threshold_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202406/ImageSetsOcrLabelNoAug_Texas/ImageSets/Main/all.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_Texas_all_result.csv')
    args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_us_202406/ImageSetsOcrLabelNoAug_Texas/ImageSets/Main/all_threshold.txt"
    args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_us_Texas_all_threshold_result.csv')
    model_test(args, bool_caffe=True)

    # # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/brazil_02210_202301/Images"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_original_result.csv')
    # model_test(args, from_jpg_dir=True, bool_caffe=True)

    # #######
    # # chile
    # #######
    # # data + diffste(200,000)
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chile_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202312_w_diffste_w_aug/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chile_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202312_w_diffste_w_aug/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chile_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202312_w_diffste_w_aug/inference/chile_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chile_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_128_256_202312_w_diffste_w_aug/inference/"
    # args.img_shape = (1, 128, 256)

    # # ocr_test
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_chile_202312/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_chile_test_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_chile_202312/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/trainval.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_chile_trainval_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_chile_202312/ImageSetsOcrLabelNoAug/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_chile_wdiffste_test_result.csv')
    # model_test(args, bool_caffe=True)

    # # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/brazil_02210_202301/Images"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_original_result.csv')
    # # model_test(args, from_jpg_dir=True, bool_caffe=True)

    # #######
    # # eu
    # #######
    # # data + diffste(200,000)
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_augg/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_augg/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_augg/inference/eu_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_eu_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20240321_w_diffste_w_augg/inference/"
    # args.img_shape = (1, 64, 256)

    # # ocr_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_eu_202403A/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_eu_test_result.csv')
    # # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_eu_202403A/ImageSetsOcrLabelNoDiffsteNoAug/ImageSets/Main/trainval.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_eu_trainval_result.csv')
    # model_test(args, bool_caffe=True)

    # # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/brazil_02210_202301/Images"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_brazil_original_result.csv')
    # # model_test(args, from_jpg_dir=True, bool_caffe=True)
