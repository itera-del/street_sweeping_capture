import argparse
import cv2
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.json.platform_json_write import PlatformJsonWriter


def platform_json_check(args):

    json_list = np.array(os.listdir(args.platform_json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]
    json_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    platform_json_writer = PlatformJsonWriter()

    check_num = 0
    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.platform_json_dir, json_name)

        # jpg
        jpg_name = str(json_name)[:-5] + '.jpg'
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if jpg_name not in jpg_list:
            jpg_name = str(json_name).replace('.json', '.jpg')
            if jpg_name not in jpg_list:
                print("[ERROR]: unknow: {}".format(jpg_name))
                continue

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        img = cv2.imread(jpg_path)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img_shape = img.shape
        
        if img_width == annotation['width'] and img_height == annotation['height']:
            continue
        
        check_num += 1
        print("{}: {}".format(check_num, json_path))

        # 标签检测和标签转换
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

                bndbox = []             
                bndbox.extend([x1, y1, x2, y2])
                bndbox.extend([label])
                rect_list.append(bndbox)
            else:
                raise NotImplemented

        out_json_path = json_path
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1w/") 
    parser.add_argument('--jpg_name', type=str, default="JPEGImages/") 
    parser.add_argument('--json_name', type=str, default="Jsons/") 
    args = parser.parse_args()
    
    print("platform json check.")
    print("input_dir: {}".format(args.input_dir))

    args.jpg_dir = os.path.join(args.input_dir, args.jpg_name)
    args.platform_json_dir = os.path.join(args.input_dir, args.json_name)
    
    platform_json_check(args)