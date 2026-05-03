import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def avi_merge(args):

    os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

    # video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(args.output_video_path, fourcc, int(args.frame), args.video_shape, True)

    cap_1 = cv2.VideoCapture(args.video_1) 
    print(int(cap_1.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
    print(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
    print(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
    print(cap_1.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

    cap_2 = cv2.VideoCapture(args.video_2) 
    print(int(cap_2.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
    print(cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
    print(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
    print(cap_2.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

    frame_1 = 0
    frame_2 = 0
    while True:

        ret_1, img_1 = cap_1.read()
        frame_1 += 1
        
        if frame_1 >= args.after_frame:

            ret_2, img_2 = cap_2.read()
            frame_2 += 1

            if not ret_1 and not ret_2: # if the camera over return false
                break
            
            if frame_1 % args.step_frame == 0:
                # img_merge
                img_merge = np.zeros(args.output_video_shape, np.uint8)

                if not ret_1:
                    img_1 = np.zeros(args.output_video_shape, np.uint8)
                if not ret_2:
                    img_2 = np.zeros(args.output_video_shape, np.uint8)

                img_1 = cv2.resize(img_1, args.video_resize_shape)
                img_2 = cv2.resize(img_2, args.video_resize_shape)
                
                img_merge[:, :int(args.video_shape[0]/2),:] = img_1
                img_merge[:, int(args.video_shape[0]/2):,:] = img_2
                output_video.write(img_merge)
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    avi_dir = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/h264_vis_replace_10G_single_head_distance_1215A/vis/"
    avi_list = np.array(os.listdir(avi_dir))

    for idx in range(len(avi_list)):

        args.avi_name = avi_list[idx]
        print(args.avi_name)

        args.video_1 = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/h264_vis_replace_10G_single_head_distance_1215A/vis/{}".format(args.avi_name)
        args.video_2 = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/h264_vis_replace_10G_single_head_distance_1216A_test/vis/{}".format(args.avi_name)
        args.output_video_path = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/merge_onlY_head_distance/{}".format(args.avi_name)
        args.output_video_shape = (1920, 4280, 3)
        args.video_shape = (4280, 1920)
        args.video_resize_shape = (2140, 1920)
        args.suffix = '.avi'
        args.after_frame = 0
        args.step_frame = 1
        args.frame = 20

        avi_merge(args)

    # args.avi_name = "CH1-251111-171818-171843.avi"
    # args.video_1 = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/h264_vis_replace_10G_single_head_distance_1212B/vis/{}".format(args.avi_name)
    # args.video_2 = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/h264_vis_replace_10G_single_head_distance_1215A/vis/{}".format(args.avi_name)
    # args.output_video_path = "/yuanhuan/data/image/AEB/AEBS_BYD_ACC/ACC_专题优化/error/B12E06_20251111_20VS5_child/merge/{}".format(args.avi_name)
    # args.output_video_shape = (1920, 4280, 3)
    # args.video_shape = (4280, 1920)
    # args.video_resize_shape = (2140, 1920)
    # args.suffix = '.avi'
    # args.after_frame = 0
    # args.step_frame = 1
    # args.frame = 20

    # avi_merge(args)
