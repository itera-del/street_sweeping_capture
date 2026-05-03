import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
import sys
from tqdm import tqdm


def jpg_to_avi(args):
    
    # mkdir 
    os.makedirs(args.output_video_dir, exist_ok=True)

    # folder list
    folder_list = os.listdir(args.input_dir)

    for idx in range(len(folder_list)):
        
        folder_name = folder_list[idx]
        input_folder = os.path.join(args.input_dir, folder_name)
        # img
        img_list = np.array(os.listdir(input_folder))
        img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
        img_list.sort()

        # video
        video_path = os.path.join(args.output_video_dir, '{}.avi'.format(folder_name))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(video_path, fourcc, 5.0, args.video_shape, True)

        for idy in tqdm(range(len(img_list))):
            img_name = img_list[idy]
            img_path = os.path.join(input_folder, img_name)

            # img
            img = cv2.imread(img_path)

            output_video.write(img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/zhangye/model/image/mm_grounding_dino/baby_carriage_pred_res/JPEGImages"
    args.output_video_dir = "/zhangye/model/image/mm_grounding_dino/baby_carriage_pred_res/avi/"
    args.res_name = "baby_carriage_pred_res"
    args.suffix = '.jpg'
    args.video_shape = (1920, 1080)

    jpg_to_avi(args)