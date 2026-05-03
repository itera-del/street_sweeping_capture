import argparse
import cv2
from collections import OrderedDict
import numpy as np
import os
from prettytable import PrettyTable
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.analysis_result.analysis_seg_mask.metrics import eval_metrics

# ###############################################
# # dataset_zd_dict
# ###############################################
# from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict import *

###############################################
# dataset_zd_dict_color
###############################################
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_color import *


def analysis_result(args):
    # img list, seg label list
    img_list = []
    seg_label_list = []

    with open(args.input_test_file_path) as f:
        for line in f:
            l = line.strip().split(" ")
            img_list.append(l[0])   
            seg_label_list.append(l[1])

    # load mask
    gt_seg_maps = []
    pred_seg_maps = []

    for idx in tqdm(range(len(seg_label_list))):

        gt_seg_path = seg_label_list[idx]
        gt_seg_name = os.path.basename(gt_seg_path)
        pred_seg_path = os.path.join(args.mask_pred_dir, gt_seg_name)

        gt_seg = cv2.imread(gt_seg_path)[:, :, 0]
        gt_seg = cv2.resize(gt_seg, (128, 64))
        pred_seg = cv2.imread(pred_seg_path)[:, :, 0]

        gt_seg_maps.append(gt_seg)
        pred_seg_maps.append(pred_seg)
    
    gt_seg_maps = np.array(gt_seg_maps)
    pred_seg_maps = np.array(pred_seg_maps)

    # test a list of files
    num_classes = len(class_seg_label)
    ignore_index = 255

    ret_metrics = eval_metrics(
        pred_seg_maps,
        gt_seg_maps,
        num_classes,
        ignore_index,
        args.metric_list,
        label_map=dict(),
        reduce_zero_label=False)

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    ret_metrics_class.update({'Class': class_seg_label})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print(class_table_data)
    print(summary_table_data)

    with open(args.mask_res_class_table_csv_path, "w") as f:
        for key, val in ret_metrics_class.items():
            f.write(key)
            f.write(",")
            val = [str(idx) for idx in val]
            f.write(",".join(val))
            f.write("\n")

    with open(args.mask_res_summary_table_csv_path, "w") as f:
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                f.write(key)
                f.write(",")
                f.write(str(val))
                f.write("\n")
            else:
                f.write('m' + key)
                f.write(",")
                f.write(str(val))
                f.write("\n")


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ###############################################
    # dataset_zd_dict
    ###############################################

    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019"

    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/"
    # args.dataset_name = "kind_num_city_cartype_merge_test"
    # args.mode = "test"
    # args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))


    # args.mask_pred_dir = os.path.join(args.output_dir, '{}/{}/mask'.format(args.mode, args.dataset_name))
    # args.mask_res_class_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_class_table.csv'.format(args.mode, args.dataset_name))
    # args.mask_res_summary_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_summary_table.csv'.format(args.mode, args.dataset_name))
    # args.metric_list = ['mIoU', 'mDice', 'mFscore']

    # analysis_result(args)


    ###############################################
    # dataset_zd_dict_color
    ###############################################

    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1101/"

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/"
    args.dataset_name = "color_merge_test"
    args.mode = "test"
    args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))


    args.mask_pred_dir = os.path.join(args.output_dir, '{}/{}/mask'.format(args.mode, args.dataset_name))
    args.mask_res_class_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_class_table.csv'.format(args.mode, args.dataset_name))
    args.mask_res_summary_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_summary_table.csv'.format(args.mode, args.dataset_name))
    args.metric_list = ['mIoU', 'mDice', 'mFscore']

    analysis_result(args)

if __name__ == '__main__':
    main()