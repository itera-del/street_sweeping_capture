from src.util import safe_make_dirs, clear_path, draw_detection_result
# from ssd_detector.mmdetection_model import MmdetModel
from ssd_detector.plate_regression import PlateRegression
from ssd_detector.ssd_vgg_fpn import SSDDetector

import cv2
import os
import torch
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def list_dirs(path, recursive=True):
    out_files = []
    if recursive:
        for file_dir, folders, files in os.walk(path):
            for file in files:
                out_files.append(os.path.join(file_dir, file))
    else:
        for file in os.listdir(path):
            out_files.append(os.path.join(path, file))
    return out_files


def get_files(paths, target='.xml'):
    out_files = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        files = list_dirs(path)
        files = filter(lambda s: os.path.splitext(s)[1] == target, files)
        out_files.extend(files)
    return out_files


def video_detect(video_path, result_video_path, car_detector, plate_detector):
    out_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(result_video_path, os.path.splitext(video_path.split('/')[-1])[0] + ".mp4"),
                          fourcc, 25.0, out_size)
    cap = cv2.VideoCapture(video_path)
    print("video_path:", video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=video_length) as p_bar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_h, frame_w, _ = frame.shape
            bboxes = car_detector.detect(frame)
            bboxes["plate"] = plate_detector.detect(frame, bboxes['car'])
            #### draw plates only
            plates = {}
            plates['plate'] = []
            for bbox in bboxes["plate"]:
                xmin, ymin, xmax, ymax = bbox
                height = ymax-ymin
                rate = (xmax-xmin)/height
                if rate >= 2.8 and rate <= 3.8 and height > 20:
                    plates['plate'].append(bbox)
            img = draw_detection_result(frame, plates, mode='ltrb')
            # print(bboxes)
            p_bar.update(1)
            out.write(img)
    cap.release()
    out.release()


def inference_video(folder_name, video_path, pt_path, out_root):
    result_path = out_root + "/"
    result_path += "{}/{}".format(folder_name, pt_path.split('/')[-1].split(".")[0])
    clear_path(result_path)
    safe_make_dirs(result_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # car_detector = MmdetModel(device=device)
    car_detector = SSDDetector(device=device)
    plate_detector = PlateRegression(pt_path, device)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(result_video_path, fourcc, 25.0, (1920, 1080))
    if not isinstance(video_path, list):
        video_path = [video_path]
    for j, v_path in enumerate(video_path):
        video_detect(v_path, result_path, car_detector, plate_detector)


def main():
    pt_path = "ssd_detector/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best.pt"
    folder_name = "video_result"
    # video_path = "/home/jhwen/workspace/data/video/singapore/1_01.avi"
    out_root = "/home/lirui/HDD/lirui/plate_regression/videos/output"
    video_path = get_files("/home/lirui/HDD/lirui/plate_regression/videos/input", ".mp4")
    inference_video(folder_name, video_path, pt_path, out_root)


if __name__ == '__main__':
    main()


