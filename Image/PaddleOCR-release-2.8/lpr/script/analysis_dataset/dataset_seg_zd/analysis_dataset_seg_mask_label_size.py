import argparse
import cv2
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_ocr_zd.dataset_mask.gen_ocr_img_augment import mask_2_bbox


def plot_hist(data, bins, xlabel='', ylabel='', title='', savefig=''):
    # 绘制直方图
    plt.hist(x = data, # 指定绘图数据
            bins = bins, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'steelblue' # 指定直方图的边框色
            )
    # plt.xlim((40, 70))
    # 添加x轴和y轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 添加标题
    plt.title(title)
    # 显示图形
    # plt.show()
    plt.savefig(savefig, dpi=300)
    plt.close()


def analysis_label_size(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_img_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    # init
    analysis_label_result_dict = {}

    for idx in tqdm(range(len(jpg_list))):

        img_name = jpg_list[idx]
        img_path = os.path.join(args.input_img_dir, img_name)
        mask_path = os.path.join(args.input_mask_dir, img_name.replace(".jpg", ".png"))
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        # img
        img = cv2.imread(img_path)
        img_size = img.shape[0] * img.shape[1]

        # mask
        mask = cv2.imread(mask_path)
        mask_size = mask.shape[0] * mask.shape[1]
        assert img_size == mask_size

        for key in ['img_height', 'img_width']:
            if key not in analysis_label_result_dict:
                analysis_label_result_dict[key] = []
            
            if key == 'img_height':
                analysis_label_result_dict[key].append(img.shape[0])
            elif key == 'img_width':
                analysis_label_result_dict[key].append(img.shape[1])

        for key in dataset_dict.class_seg_label_group_2_id_map.keys():

            analysis_label_list = dataset_dict.class_seg_label_group_2_id_map[key]
            analysis_label_mask = np.array((0,0))
            analysis_label_max_size = 0.0

            for idy in range(len(analysis_label_list)): 
                analysis_label = analysis_label_list[idy]
                analysis_label_size_idy = ((mask == dataset_dict.id_2_mask_id_dict[analysis_label])[:,:,0]).sum()

                if analysis_label_size_idy > analysis_label_max_size:
                    analysis_label_mask = (mask == dataset_dict.id_2_mask_id_dict[analysis_label])[:,:,0].astype(np.uint8)
                    analysis_label_max_size = analysis_label_size_idy
            
            if analysis_label_max_size > 0.0:
                
                if key not in analysis_label_result_dict:
                    analysis_label_result_dict[key] = []
                if key + "_height" not in analysis_label_result_dict:
                    analysis_label_result_dict[key + "_height"] = []
                if key + "_width" not in analysis_label_result_dict:
                    analysis_label_result_dict[key + "_width"] = []

                analysis_label_result_dict[key].append( float("{:.4f}".format(float(analysis_label_max_size) / float(img_size))) )

                analysis_label_bbox = mask_2_bbox(analysis_label_mask)
                analysis_label_result_dict[key + "_height"].append(analysis_label_bbox[0][3])
                analysis_label_result_dict[key + "_width"].append(analysis_label_bbox[0][2])

    print( analysis_label_result_dict )

    for key in analysis_label_result_dict.keys():

        print( "{}: ".format(key), "max: ", np.array(analysis_label_result_dict[key]).max(), ", min: ", np.array(analysis_label_result_dict[key]).min())
        plot_hist(np.array(analysis_label_result_dict[key]), 100, '{}'.format(key), 'frequency', 'Hist For {}'.format(key), \
                    os.path.join(args.output_dir, "hist_for_{}.png".format(key))) 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop_0804_0809/") 
    args = parser.parse_args()

    print("2、analysis dataset seg mask label size.")
    print("input_dir: {}".format(args.input_dir))

    args.seg_dict_name = "dataset_zd_dict"
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.input_mask_dir = os.path.join(args.input_dir, "city_label/mask")
    args.output_dir = os.path.join(args.input_dir, "analysis")

    analysis_label_size(args)
