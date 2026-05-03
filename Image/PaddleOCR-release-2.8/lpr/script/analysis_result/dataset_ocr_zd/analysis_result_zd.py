import argparse
import os
from tkinter.messagebox import NO
import pandas as pd
from prettytable import PrettyTable
import sys
import sklearn.metrics

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.metrics_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_nomal import *


def analysis_total(args):

    print("**********************************")
    print("** {}".format("total"))
    print("**********************************")

    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)

    # init
    true_list = []
    pred_list = []

    for _, row in csv_pd.iterrows():
        true_list.append(1)
        pred_list.append(int(row['res']))

    tn, fp, fn, tp = get_confusion_matrix(true_list, pred_list)
    accuracy = get_accuracy(tn, fp, fn, tp)
    res_dict = {
                    'class': "all", 
                    'acc': "{:.2f}%".format(accuracy * 100), 
                    'num': len(true_list),
                }
    print(res_dict)

    return [res_dict]


def analysis_event(args, event, event_name_list):

    print("**********************************")
    print("** {}".format(event))
    print("**********************************")

    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)
    
    # init 
    res_list = []               # [ {'class': class, 'acc': tpr, 'num': num } ]

    for idx in range(len(event_name_list)):

        analysis_idx = event_name_list[idx]
        analysis_csv_pd = csv_pd[csv_pd[event] == analysis_idx]

        if not len(analysis_csv_pd):
            continue

        # init
        true_list = []
        pred_list = []

        for _, row in analysis_csv_pd.iterrows():
            true_list.append(1)
            pred_list.append(int(row['res']))

        tn, fp, fn, tp = get_confusion_matrix(true_list, pred_list)
        accuracy = get_accuracy(tn, fp, fn, tp)
        res_dict = {
                        'class': "{}_{}".format(event, analysis_idx),
                        'acc': "{:.2f}%".format(accuracy * 100), 
                        'num': len(true_list),
                    }
        
        res_list.append(res_dict)
    
    print(res_list)

    return res_list


def analysis_result(args):
    
    # init 
    res_list = []

    res_list.extend(analysis_total(args))

    if args.analysis_event_bool:
        for key in args.analysis_event_dict.keys():
            res_list.extend(analysis_event(args, key, args.analysis_event_dict[key]))

    # out csv
    error_pd = pd.DataFrame(res_list)
    error_pd.to_csv(args.out_csv_path, index=False, encoding="utf_8_sig")


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_hisi_1010/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1111/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_1120/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230119/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_20230217/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230209/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_tcres_20230217/"
    args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230301/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v3_en_mobile/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v2_en_mobile_pp-OCRv2/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_no_pretrain/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_tc_res_mobile/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_cnn/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_dnn/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_rnn/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_enhanced_ctc/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_distillation/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_distillation/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_gray/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresizeratio_gray/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_all/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_20230119_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_white_gray_64_320_1220_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_end_64_320_1222_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_white_gray_end_64_320_1220_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_end_64_384_1226_all_aug/best_accuracy/"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_white_gray_end_64_384_1220_all_aug/best_accuracy/"

    # ocr_merge_test
    args.input_csv_path = os.path.join(args.input_dir, 'test/ocr_merge_test_result.csv')
    args.out_csv_path = os.path.join(args.input_dir, "test/analysis_ocr_merge_test_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'test_onnx/ocr_merge_test_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "test_onnx/analysis_ocr_merge_test_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'test_caffe/ocr_merge_test_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "test_caffe/analysis_ocr_merge_test_result.csv")

    # # args.input_csv_path = os.path.join(args.input_dir, 'test/ocr_merge_test_beam_search_result.csv')
    # # args.out_csv_path = os.path.join(args.input_dir, "test/analysis_ocr_merge_test_beam_search_result.csv")
    # args.analysis_event_bool = True
    args.analysis_event_bool = False
    
    # # from_jpg_dir
    # args.input_csv_path = os.path.join(args.input_dir, 'test/data_synthesis_baozihao_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "test/analysis_data_synthesis_baozihao_result.csv")
    # # args.input_csv_path = os.path.join(args.input_dir, 'test_onnx/data_synthesis_baozihao_result.csv')
    # # args.out_csv_path = os.path.join(args.input_dir, "test_onnx/analysis_data_synthesis_baozihao_result.csv")
    # # args.input_csv_path = os.path.join(args.input_dir, 'test_caffe/data_synthesis_baozihao_result.csv')
    # # args.out_csv_path = os.path.join(args.input_dir, "test_caffe/analysis_data_synthesis_baozihao_result.csv")
    # # args.analysis_event_bool = True
    # args.analysis_event_bool = False

    args.analysis_event_dict = {
                                'country': country_name_list, 
                                'city': city_name_list, 
                                'color': [str(color_idx).lower() for color_idx in color_name_list],
                                'column': column_name_list,
                                }

    analysis_result(args)


if __name__ == '__main__':
    main()