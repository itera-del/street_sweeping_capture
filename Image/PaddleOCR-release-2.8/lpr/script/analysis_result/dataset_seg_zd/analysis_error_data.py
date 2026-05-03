import argparse
import importlib
import os
import pandas as pd
import shutil
import sys 
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')


def analysis_event(args, event):
    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)

    for _, row in tqdm(csv_pd.iterrows(), total=len(csv_pd)):

        bbox_img_name = str(os.path.basename(row['file']))
        mask_img_name = str(os.path.basename(row['file'])).replace('.jpg', '.png')
        label = row['{}'.format(event)]
        pred = row['{}_res'.format(event)]

        # city
        sub_folder_name = "{}--{}".format(label, pred)
        output_folder = os.path.join(args.output_analysis_data_dir, '{}'.format(event), sub_folder_name)
        create_folder(output_folder)

        input_path = os.path.join(args.bbox_img_dir, bbox_img_name)
        output_path = os.path.join(output_folder, bbox_img_name)
        shutil.copy(input_path, output_path)


def analysis_result(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name)

    # mkdir 
    create_folder(args.output_analysis_data_dir)

    for key in dataset_dict.class_seg_label_group_2_name_map.keys():
        analysis_event(args, key)


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # ##############################################
    # # dataset_zd_dict
    # ##############################################

    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019"

    # args.seg_dict_name = "dataset_zd_dict"
    # args.dataset_name = "merge_test_0804_0809_0810_0811"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.bbox_img_dir = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))
    # args.output_analysis_data_dir = os.path.join(args.output_dir, '{}/{}/analysis_data'.format(args.mode, args.dataset_name))

    # analysis_result(args)

    # ###############################################
    # # dataset_zd_dict_color
    # ###############################################

    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1101/"

    # args.seg_dict_name = "dataset_zd_dict_color"
    # args.dataset_name = "merge_test_0804_0809_0810_0811"
    # args.mode = "test"
    # # args.mode = "trainval"
    # args.input_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    # args.bbox_img_dir = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))
    # args.output_analysis_data_dir = os.path.join(args.output_dir, '{}/{}/analysis_data'.format(args.mode, args.dataset_name))

    # analysis_result(args)

    ###############################################
    # dataset_zd_dict & dataset_zd_dict_color
    ###############################################

    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1103/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_zd_1104/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1107/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_no_grad_zd_1108/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_1210/"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230119/"

    args.seg_dict_name = "dataset_zd_dict"
    # args.seg_dict_name = "dataset_zd_dict_color"
    args.dataset_name = "ImageSetsCityColorLabelNoAug"
    args.mode = "test"
    # args.mode = "trainval"
    args.input_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    args.bbox_img_dir = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))
    args.output_analysis_data_dir = os.path.join(args.output_dir, '{}/{}/analysis_data'.format(args.mode, args.dataset_name))

    analysis_result(args)


if __name__ == '__main__':
    main()