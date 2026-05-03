# 各个数据在墨西哥字典中的占比白名单
import argparse
import cv2
import os
import pandas as pd
import re
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_huanwei.dataset_dict.dataset_huanwei_dict_normal import *


def analysis_dataset_label(args):

    # mkdir
    create_folder(args.output_analysis_dir)

    # init
    analysis_dict = {}

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        
        # 【修改点】：直接转换为字符串，不限制长度，不自动补零
        plate_num = str(row['num'])

        # analysis info
        analysis_info = {}
        analysis_info["num"] = plate_num

        for key_name in analysis_label_columns.keys():
            
            for analysis_key in analysis_info.keys():

                if key_name != analysis_key:
                    continue
                
                if key_name == "num":
                    num = analysis_info[analysis_key]

                    for num_idx in range(len(num)):

                        if num[num_idx] not in analysis_label_columns[key_name]:
                            # 建议优化：抛出带具体信息的异常，方便排查脏数据
                            raise ValueError(f"非法字符: '{num[num_idx]}' 位于图片 {img_path}")

                        if key_name not in analysis_dict:
                            analysis_dict[key_name] = {}
                        
                        if num[num_idx] not in analysis_dict[key_name]:
                            analysis_dict[key_name][num[num_idx]] = 1
                        else:
                            analysis_dict[key_name][num[num_idx]] += 1

                else:

                    if analysis_info[analysis_key] not in analysis_label_columns[key_name]:
                        raise Exception

                    if key_name not in analysis_dict:
                        analysis_dict[key_name] = {}
                    
                    if analysis_info[analysis_key] not in analysis_dict[key_name]:
                        analysis_dict[key_name][analysis_info[analysis_key]] = 1
                    else:
                        analysis_dict[key_name][analysis_info[analysis_key]] += 1

    # write
    for key_name in analysis_label_columns.keys():
    
        if key_name not in analysis_dict:
            continue

        out_csv_path = os.path.join(args.output_analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name], index=[args.date_name], columns=analysis_label_columns[key_name])
        analysis_pd.to_csv(out_csv_path, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Mexican_20260331_frames_5f_10060")
    parser.add_argument('--input_csv_dir', type=str, default="/defaultShare/mexican_lpr/Mexican_20260331_frames_5f_10060/out_csv_0414")
    parser.add_argument('--output_csv_dir', type=str, default="",
                        help="兼容 dataset_csv_mexican.py 的输出参数，等价于 input_csv_dir")
    parser.add_argument('--input_csv_path', type=str, default="")
    parser.add_argument('--output_analysis_dir', type=str, default="")
    parser.add_argument('--bool_append_date_name_dir', action='store_true', default=False)
    args = parser.parse_args()

    # 兼容上一条脚本参数名：--output_csv_dir
    if args.output_csv_dir:
        args.input_csv_dir = args.output_csv_dir

    if args.input_csv_path == "":
        args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')

    if args.output_analysis_dir == "":
        # 当输入目录形如 out_csv_0414 时，自动映射为 analysis_0414
        input_dir_name = os.path.basename(os.path.normpath(args.input_csv_dir))
        output_dir_name = re.sub(r'^out_csv_', 'analysis_', input_dir_name)
        args.output_analysis_dir = os.path.join(os.path.dirname(os.path.normpath(args.input_csv_dir)), output_dir_name)

    if args.bool_append_date_name_dir:
        args.output_analysis_dir = os.path.join(args.output_analysis_dir, args.date_name)

    print("analysis dataset label num.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_analysis_dir: {}".format(args.output_analysis_dir))

    analysis_dataset_label(args)