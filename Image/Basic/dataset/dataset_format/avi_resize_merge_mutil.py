import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def avi_merge(args):

    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    avi_list = np.array(os.listdir(args.avi_dir))
    for idx in range(len(avi_list)):
        avi_name = avi_list[idx]
        print(avi_name)

        videos = []
        total_frames = []
        for idx in range(len(args.video_dir_list)):
            key = args.input_name_list[idx]
            video_path = os.path.join(args.video_dir_list[idx], avi_name)
            assert os.path.exists(video_path)
            
            vcap = cv2.VideoCapture(video_path)
            frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            videos.append((key, vcap))
            total_frames.append(frames)

        final_frames = min(total_frames)
        wtr = cv2.VideoWriter(os.path.join(args.output_dir, avi_name), cv2.VideoWriter_fourcc(*"XVID"),25.0,(1920,1080),True)
        IMW = int(1920//2.3)
        IMH = int(1080//2.3)

        for i in tqdm(range(final_frames)):

            back = np.zeros([1080,1920,3], dtype="uint8")
            for r in range(2):
                for c in range(2):
                    y1 = 1080//2*r
                    y2 = y1 + IMH
                    x1 = c * IMW
                    x2 = x1 + IMW
                    idx = r * 2 + c
                    if idx >= len(videos):
                        continue
                    key, vcap = videos[idx]
                    ret, image = vcap.read()
                    assert ret
                    image = cv2.resize(image, (IMW, IMH))
                    back[y1:y2, x1:x2, :] = image
                    cv2.putText(back, f"{key}", (x1+50, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
            
            wtr.write(back)
        
        wtr.release()
        for key, vcap in videos:
            vcap.release()
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.avi_dir = "/yuanhuan/model/image/yolov6/yolov6_c28_car_0320/dataset_检测难例收集/epoch_400/avi/"
    args.video_dir_list = ["/yuanhuan/model/image/yolov6/yolov6_c28_car_0320/dataset_检测难例收集/epoch_400/avi/",
                        #    "/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture/dataset_检测难例收集/epoch_20/avi",
                        #    "/yuanhuan/model/image/yoloworld/yolo_world_v2_l_vlpan_bn_adamw_2e-4_200e_2gpus_finetune_640_640_capture/capture_class_texts/capture_part_100_2w/dataset_检测难例收集/epoch_80/avi/",
                           "/yuanhuan/model/image/yolov6/yolov6_rm_capture_90w_512_288/dataset_检测难例收集/epoch_300/avi",
                           "/yuanhuan/model/image/yolox/yolox_rm_capture_90w_576_320/dataset_检测难例收集/epoch_300/avi",
                           "/yuanhuan/model/image/yolov6_v3_0/yolov6_v3_0_rm_capture_90w_512_288/dataset_检测难例收集/epoch_300/avi",
                           ]
    # args.avi_dir = "/yuanhuan/model/image/yolov6/yolov6_c27_car_bus_truck_moto_plate_0731/dataset_检测难例收集/epoch_260/avi/"
    # args.video_dir_list = ["/yuanhuan/model/image/yolov6/yolov6_c27_car_bus_truck_moto_plate_0731/dataset_检测难例收集/epoch_260/avi/",
    #                     #    "/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture/dataset_检测难例收集/epoch_20/avi",
    #                     #    "/yuanhuan/model/image/yoloworld/yolo_world_v2_l_vlpan_bn_adamw_2e-4_200e_2gpus_fine tune_640_640_capture/capture_class_texts/capture_part_100_2w/dataset_检测难例收集/epoch_80/avi/",
    #                        "/yuanhuan/model/image/yolov6/yolov6_rm_capture_90w_512_288/dataset_检测难例收集/epoch_300/avi",
    #                        "/yuanhuan/model/image/yolox/yolox_rm_capture_90w_576_320/dataset_检测难例收集/epoch_300/avi",
    #                        "/yuanhuan/model/image/yolov6_v3_0/yolov6_v3_0_rm_capture_90w_512_288/dataset_检测难例收集/epoch_300/avi",
    #                        ]
    # args.avi_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_车脸漏检视频/plate/avi/"
    # args.video_dir_list = ["/yuanhuan/model/image/yolov6/yolov6_c27_car_bus_truck_moto_plate_0731/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/avi",
    #                     #    "/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/avi/",
    #                     #    "/yuanhuan/model/image/yoloworld/yolo_world_v2_l_vlpan_bn_adamw_2e-4_200e_2gpus_finetune_640_640_capture/capture_class_texts/capture_part_100_2w/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/avi/",
    #                         "/yuanhuan/model/image/yolov6/yolov6_rm_capture_90w_512_288/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/avi",
    #                         "/yuanhuan/model/image/yolox/yolox_rm_capture_90w_576_320/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/epoch_300/avi",
    #                         "/yuanhuan/model/image/yolov6_v3_0/yolov6_v3_0_rm_capture_90w_512_288/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/plate/epoch_300/avi",
    #                        ]
    
    args.input_name_list = ["yolov6_o",
                            "yolov6_90w",
                            "yolox",
                            "yolov6_v3_0",
                            ]
    args.output_dir = "/yuanhuan/model/image/yolox/yolox_rm_capture_90w_576_320/dataset_检测难例收集/epoch_300/avi_res_4"
    args.output_video_shape = (1080, 1920, 3)
    
    avi_merge(args)