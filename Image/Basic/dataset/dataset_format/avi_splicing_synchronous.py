import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm


def avi_merge(args):

    # mkdir 
    os.makedirs(args.output_video_dir, exist_ok=True)

    # video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path_A = os.path.join(args.output_video_dir, args.output_video_path_A)
    output_video_A = cv2.VideoWriter(output_video_path_A, fourcc, int(args.frame), args.output_video_shape_A, True)
    output_video_path_B = os.path.join(args.output_video_dir, args.output_video_path_B)
    output_video_B = cv2.VideoWriter(output_video_path_B, fourcc, int(args.frame), args.output_video_shape_B, True)
    
    video_dir_list_A = os.listdir(args.video_dir_A)
    video_dir_list_A.sort()
    video_dir_list_B = os.listdir(args.video_dir_B)
    video_dir_list_B.sort()

    black_screen_num = 0
    
    for idx in tqdm(range(len(video_dir_list_A))):
        video_path_A = os.path.join(args.video_dir_A, video_dir_list_A[idx])
        video_path_B = os.path.join(args.video_dir_B, video_dir_list_B[idx])
        assert video_path_A[-4:] in args.suffix_list
        assert video_path_B[-4:] in args.suffix_list
        print(video_path_A)
        print(video_path_B)

        cap_A = cv2.VideoCapture(video_path_A) 
        print(int(cap_A.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap_A.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap_A.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap_A.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        cap_B = cv2.VideoCapture(video_path_B) 
        print(int(cap_B.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap_B.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap_B.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap_B.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        while True:

            ret_A, img_A = cap_A.read()
            ret_B, img_B = cap_B.read()
            
            if not ret_A and not ret_B: # if the camera over return false
                
                while black_screen_num < args.black_screen_frame:
                    black_screen_img = np.zeros_like(old_img_A)
                    output_video_A.write(black_screen_img)
                    black_screen_num += 1

                    black_screen_img = np.zeros_like(old_img_B)
                    output_video_B.write(black_screen_img)
                    black_screen_num += 1
                
                black_screen_num = 0
                break

            if ret_A:
                output_video_A.write(img_A)
                old_img_A = img_A
            else:
                black_screen_img = np.zeros_like(old_img_A)
                output_video_A.write(black_screen_img)

            if ret_B:
                output_video_B.write(img_B)
                old_img_B = img_B
            else:
                black_screen_img = np.zeros_like(old_img_B)
                output_video_B.write(black_screen_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.video_dir_A = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C28_mini/FL4-FL5"
    args.video_dir_B = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C27_mini/FL4-FL5"
    args.output_video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/功能验证测试样例/C27C28联动抓拍/FL4-FL5/"
    args.output_video_path_A = "FL4-FL5_C28.avi"
    args.output_video_path_B = "FL4-FL5_C27.avi"
    args.output_video_shape_A = (1920, 1080)
    args.output_video_shape_B = (2592, 1920)

    args.suffix_list = ['.avi', '.mp4']
    args.black_screen_frame = 10 * 25
    args.frame = 25

    avi_merge(args)