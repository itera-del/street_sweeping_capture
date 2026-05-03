import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

def avi_length(args):

    # init
    video_num = 0
    frame_num = 0

    # video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for idx in tqdm(range(len(args.video_dir_list))):
        video_dir = args.video_dir_list[idx]

        video_list = os.listdir(video_dir)
        video_list.sort()
        for idy in tqdm(range(len(video_list))):
            video_path = os.path.join(video_dir, video_list[idy])
            assert video_path[-4:] in args.suffix_list

            cap = cv2.VideoCapture(video_path) 
            print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
            print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
            print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
            print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

            video_num += 1
            frame_num += cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(video_num)
    print(frame_num/int(args.frame)/60)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜间_侧向_前方期望提升算法召回率_20241211/avi_cut"]
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/avi",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_后向_0615/avi"]
    args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_侧向_0615/avi",
                           "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/avi/00000G000170",
                           "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/avi/00000G000171"]
    
    args.suffix_list = ['.avi', '.mp4']
    args.frame = 25

    avi_length(args)