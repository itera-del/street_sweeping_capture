import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm


def avi_merge(args):

    # mkdir 
    os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

    # video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(args.output_video_path, fourcc, int(args.frame), args.output_video_shape, True)
    
    for idx in tqdm(range(len(args.video_dir_list))):
        video_dir = args.video_dir_list[idx]

        black_screen_num = 0
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
    
            while True:
    
                ret, img = cap.read()

                if not ret: # if the camera over return false
                    
                    while black_screen_num < args.black_screen_frame:
                        black_screen_img = np.zeros_like(old_img)
                        output_video.write(black_screen_img)
                        black_screen_num += 1
                    
                    black_screen_num = 0
                    break
                
                output_video.write(img)
                old_img = img


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/测试样例/安全岛场景/常规场景/avi/后向C28/",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/测试样例/城市道路场景/常规场景/avi/后向C28"]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/merge/后向C28_safeisland_常规场景测试样例.avi"
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/测试样例/安全岛场景/常规场景/avi/前向C28/",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/测试样例/城市道路场景/常规场景/avi/前向C28"]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/merge/前向C28_safeisland_常规场景测试样例.avi"
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20231211_C28误报/avi/前向C28/",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20231205_20231214_C27漏报_C28误报/C28误报/avi/前向C28",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C28_mini/FL2-FL3",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C28_mini/FL4-FL5",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_1031_1107/avi/前向C28"
    #                        ]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/merge/前向C28_merge.avi"
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜间_侧向_前方期望提升算法召回率_20241211/avi_cut"]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜间_侧向_前方期望提升算法召回率_20241211/avi_merge/C27_夜间_侧向_20241211_merge.avi"
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/avi",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_后向_0615/avi"]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/merge/5M_白天_0615_merge.avi"
    # args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_侧向_0615/avi",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/avi/00000G000170",
    #                        "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/avi/00000G000171"]
    # args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/merge/5M_夜晚_0615_merge.avi"
    args.video_dir_list = ["/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_MOROCCO_C27/2M5M_白天_0329/avi/avi_2592_1520"]
    args.output_video_path = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_MOROCCO_C27/2M5M_白天_0329/avi/avi_2592_1520_merge.avi"

    args.suffix_list = ['.avi', '.mp4']
    args.black_screen_frame = 10 * 25

    # args.output_video_shape = (2592, 1920)
    args.output_video_shape = (2592, 1520)
    # args.output_video_shape = (1920, 1080)
    # args.output_video_shape = (1280, 720)
    args.frame = 25

    avi_merge(args)