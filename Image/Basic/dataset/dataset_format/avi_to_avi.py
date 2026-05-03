import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def avi_to_avi(args):
    # mkdir 
    create_folder(args.output_video_dir)
    
    # file
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    for idx in tqdm(range(len(video_list))):
        file_name = video_list[idx]
        video_path = os.path.join(args.video_dir, file_name)

        cap = cv2.VideoCapture(video_path)
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        # video
        output_video_path = os.path.join(args.output_video_dir, '{}'.format(file_name))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), args.output_video_shape, True)

        frame_idx = 0
        
        while True:

            ret, img = cap.read()

            if not ret: # if the camera over return false
                break
            
            if frame_idx >= args.start_s * int(cap.get(cv2.CAP_PROP_FPS)):
                # img = np.fliplr(img) 
                img = cv2.resize(img, args.output_video_shape)
                output_video.write(img)
            
            if frame_idx > args.end_s * int(cap.get(cv2.CAP_PROP_FPS)):
                break

            frame_idx += 1



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜间_侧向_前方期望提升算法召回率_20241211/avi"
    args.output_video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜间_侧向_前方期望提升算法召回率_20241211/avi_cut"
    args.suffix = '.mp4'

    args.output_video_shape = (2592, 1920)
    args.start_s = 0
    args.end_s = 300

    avi_to_avi(args)