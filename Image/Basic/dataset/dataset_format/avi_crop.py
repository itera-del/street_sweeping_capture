import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

import xml.etree.ElementTree as ET


def draw_img(args, img, bndbox):
    
    # draw
    cv2.rectangle(img, (int(bndbox[0]), int(bndbox[1])), (int(bndbox[2]), int(bndbox[3])), args.label_color[0], 4)

    return 


def crop_img(args, img, bndbox):

    img = img[bndbox[1]: bndbox[3], bndbox[0]: bndbox[2], :]

    return img


def avi_crop(args):

    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    for idx in tqdm(range(len(args.input_dir_list))):

        file_path = args.input_dir_list[idx]
        file_name = os.path.basename(file_path)
        cap = cv2.VideoCapture(file_path) 

        # video
        output_video_path = os.path.join(args.output_dir, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, int(25), args.img_size, True)

        frame_idx = 0
        while True:

            ret, img = cap.read()

            if not ret: # if the camera over return false
                break         

            # output_img = draw_img(args, img, args.bndbox[idx])
            output_img = crop_img(args, img, args.bndbox[idx])
            output_img = cv2.resize(output_img, args.img_size)
            output_video.write(output_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir_list = [
                            "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi/00D201DFF2-video_2_2_2668974835895369744.mp4",
                            "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi/00D201F5A0-video_2_2_2639387597004095519.mp4",
                            "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi/00D201F6D4-video_2_2_2638784411528593438.mp4",
                            "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi/00D201F72B-video_2_2_2668534172519235614.mp4",
                            "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi/video_2_2_2647242988572377104.mp4",
    ]
    args.output_dir = "/mnt/huanyuan2/data/image/RM_Facialfatigue/eaxample_20240702/avi_crop/"
    args.img_size = (256, 256)
    args.bndbox = [[797, 132, 1053, 388], [822, 132, 1078, 388], [797, 132, 1053, 388], [782, 92, 1038, 348], [822, 142, 1078, 398]]
    args.label_color = [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31), (106, 0, 228), (60, 100, 0), (80, 100, 0)]

    avi_crop(args)



            