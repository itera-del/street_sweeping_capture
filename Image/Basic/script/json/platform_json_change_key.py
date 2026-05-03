import argparse
import cv2
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def platform_json_change_key(args):

    # mkdir 
    create_folder(args.out_platform_json_dir)

    json_list = np.array(os.listdir(args.platform_json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.platform_json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace('.json', '.jpg')
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if jpg_name not in jpg_list:
            print("[ERROR]: unknow: {}".format(jpg_name))
            continue

        # xml 
        out_json_path = os.path.join(args.out_platform_json_dir, json_name)

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        img_width = annotation['width']
        img_height = annotation['height']
        img_shape = np.array([img_height, img_width, 3])

        rect_list = []
        for track in annotation['shapes']:
            label = track['label']
            type = track['type']

            if type == 'rectangle':
                # +1 的目的：更换了标注工具，保证 xml 结果统一：
                # ssd rfb 代码：cur_pt = int(float(bbox.find(pt).text))
                points = np.array(track['points'])
                x1 = max(int(points[0]), 1) 
                y1 = max(int(points[1]), 1) 
                x2 = min(int(points[2]), img_width - 1)
                y2 = min(int(points[3]), img_height - 1)

                if label in args.det_dict.keys():
                    label = args.det_dict[label]
                if label not in args.det_all:
                    print(label)
                    raise NotImplemented
                rect_list.append([x1, y1, x2, y2, label])

            elif type == 'polygon':
                points = np.array(track['points'])

                points_list = []
                for x, y in zip(points[::2], points[1::2]):
                    points_list.append([x, y])
                
                points_list = np.array(points_list)
                x1 = max(int(min(points_list[:, 0])) + 1, 1)
                y1 = max(int(min(points_list[:, 1])) + 1, 1)
                x2 = min(int(max(points_list[:, 0])) + 1, img_width - 1)
                y2 = min(int(max(points_list[:, 1])) + 1, img_height - 1)

                if label in args.det_dict.keys():
                    label = args.det_dict[label]
                if label not in args.det_all:
                    print(label)
                    raise NotImplemented
                rect_list.append([x1, y1, x2, y2, label])

        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_Capture/original_dupes_0_1_0_95") 
    parser.add_argument('--jpg_name', type=str, default="JPEGImages/") 
    parser.add_argument('--json_name', type=str, default="Json/") 
    parser.add_argument('--out_json_name', type=str, default="Json_refine/") 
    args = parser.parse_args()
    
    print("platform json to xml.")
    print("input_dir: {}".format(args.input_dir))

    args.jpg_dir = os.path.join(args.input_dir, args.jpg_name)
    args.platform_json_dir = os.path.join(args.input_dir, args.json_name)
    args.out_platform_json_dir = os.path.join(args.input_dir, args.out_json_name)

    args.det_dict = {
        'bicycle': 'bicyclist',
        'motorcycle': 'motorcyclist',
    }
    args.det_all=['car', 'bus', 'truck', 'bicyclist', 'motorcyclist', 'license']
    
    platform_json_change_key(args)