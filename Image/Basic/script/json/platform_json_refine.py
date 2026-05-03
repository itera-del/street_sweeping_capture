import argparse
import numpy as np
import json
import os
from tqdm import tqdm

import sys
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.json.platform_json_write import PlatformJsonWriter


def json_refine(args):

    os.makedirs(args.output_json_dir, exist_ok=True)

    json_list = os.listdir(args.json_dir)
    for idx in tqdm(range(len(json_list))):
        
        json_bboxes = {}
        json_path = os.path.join(args.json_dir, json_list[idx])
        output_json_path = os.path.join(args.output_json_dir, json_list[idx])

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

            for track in annotation['shapes']:
                label = track['label']
                points = track['points']
                type = track['type']
                if type != "polygon":
                    continue

                contours = []
                for x, y in zip(points[::2], points[1::2]):
                    contours.append([int(x), int(y)])

                step_num = int(len(contours) / args.step)
                points_list = []
                if step_num >= 1:
                    contours_step = contours[::step_num]
                    
                    for idy in range(len(contours_step)):
                        point_idx = contours_step[idy]
                        points_list.append(float(point_idx[0]))
                        points_list.append(float(point_idx[1]))
                else:
                    for idy in range(len(contours)):
                        point_idx = contours[idy]
                        points_list.append(float(point_idx[0]))
                        points_list.append(float(point_idx[1]))

                json_bboxes[label] = []     
                json_bboxes[label].append(points_list)

        # 多边形至少有3个点
        if label in json_bboxes and len(json_bboxes[label][0]) >= 6:
            platform_json_writer = PlatformJsonWriter()
            platform_json_writer.write_json(annotation['width'], annotation['height'], annotation['img_name'], output_json_path, frame_num=annotation['frame_num'], polygon_dict=json_bboxes)
        else:
            platform_json_writer = PlatformJsonWriter()
            platform_json_writer.write_json(annotation['width'], annotation['height'], annotation['img_name'], output_json_path, frame_num=annotation['frame_num'])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="zhongdong_c28_detection") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C28_safeisland/original_road_seg/") 
    parser.add_argument('--json_folder', type=str, default="refine/Jsons") 
    parser.add_argument('--output_json_folder', type=str, default="refine/Jsons_refine") 
    args = parser.parse_args()
    
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.output_json_dir = os.path.join(args.input_dir, args.date_name, args.output_json_folder)
    args.step = 30
    json_refine(args)