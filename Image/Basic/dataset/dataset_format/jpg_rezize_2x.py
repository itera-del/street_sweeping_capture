import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm


def jpg_rezize_2x(args):

    os.makedirs(args.output_dir, exist_ok=True)

    # jpg list
    jpg_list = np.array(os.listdir(args.input_dir))

    for idx in tqdm(range(len(jpg_list))):
        # jpg
        jpg_name = jpg_list[idx]
        jpg_path = os.path.join(args.input_dir, jpg_name)

        # output
        output_path = os.path.join(args.output_dir, jpg_name)

        img = cv2.imread(jpg_path)
        # img = cv2.resize(img, (int(img.shape[1] * 4), int(img.shape[0] * 4)))
        img = cv2.resize(img, (int(args.resize_shape[1]), int(args.resize_shape[0])))
        cv2.imwrite(output_path, img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/liangfang_test"
    args.output_dir = "/yuanhuan/liangfang_test"
    args.resize_shape = [512, 256]

    jpg_rezize_2x(args)
