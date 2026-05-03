import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')
from script.lpr.dataset.dataset_cn.dataset_train_test_split.data_train_test_split_seg_merge import split


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr_name', type=str, default="plate_huanwei_20240914") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_HUANWEI/training_ocr/")   
    parser.add_argument('--diffste', action='store_true', default=False)   
    args = parser.parse_args()
    
    args.input_dir = os.path.join(args.input_dir, args.ocr_name)

    print("data train test split merge.")
    print("input_dir: {}".format(args.input_dir))

    if args.diffste == True:
        args.ignore_list = [
                            'ImageSetsOcrLabelNoAug', 
                            'ImageSetsOcrLabel', 
                            'ImageSetsOcrLabelNoDiffsteNoAug',
                            'ImageSetsOcrLabelNoDiffste',
                            ]
    else:
        args.ignore_list = [
                            'ImageSetsOcrLabelNoAug', 
                            'ImageSetsOcrLabel', 
                            'ImageSetsOcrLabelNoDiffsteNoAug',
                            'ImageSetsOcrLabelNoDiffste',
                            'diffste_huanwei_20250820_20250925_3w',
                            'diffste_huanwei_20240910_20240911_1018_60w',
                            'diffste_huanwei_test_40w',
                            'diffste_huanwei_20240912_323_20w',
                            ]
        
    args.merge_file_list = [
                            "ImageSets/Main/trainval.txt", 
                            "ImageSets/Main/train.txt",
                            "ImageSets/Main/val.txt",
                            "ImageSets/Main/test.txt",
                            ]

    dir_list = np.array(os.listdir(args.input_dir))
    dir_list = list(dir_list[[dir not in args.ignore_list for dir in dir_list]])
    dir_list = [os.path.join(args.input_dir, dir) for dir in dir_list]   
    args.merge_dir_list = dir_list

    ###############################################
    # dataset_cn_dict
    ###############################################
    ## no aug
    ###############################################
    args.label_name = ""
    args.bool_aug = False
    if args.diffste == True:
        args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabelNoAug")
    else:
        args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabelNoDiffsteNoAug")

    split(args)

    ###############################################
    ## with aug 
    ###############################################
    args.label_name = ""
    args.bool_aug = True
    if args.diffste == True:
        args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabel")
    else:
        args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabelNoDiffste")

    split(args)
