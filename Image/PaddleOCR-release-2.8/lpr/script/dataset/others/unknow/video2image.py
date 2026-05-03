import os
import cv2
import tqdm
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input'     , type=str  , default='./input/dataset_bsdseg/video/1.avi'   , help='input video file')
    parser.add_argument('--output'    , type=str  , default='./output/bsd/image/1/'   , help='output image path')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    video_reader = cv2.VideoCapture(args.input)
    num, fps = int(video_reader.get(7)), int(video_reader.get(5))
    hei, wid = int(video_reader.get(4)), int(video_reader.get(3))
    for ii in range(num):
        (flag, image) = video_reader.read()
        cv2.imwrite(args.output+str(ii)+".jpg", image)