import os
import time
import cv2
import numpy as np

def buildDir(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def get_time_str(minute_level=False):
    # ex:'2018_03_30_20_34'
    if minute_level:
        return time.strftime("%Y_%m_%d_%H_%M", time.localtime((time.time())))
    else:
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime((time.time())))


def list_dirs(path):
    out_files = []
    for file_dir, folders, files in os.walk(path, followlinks=True):
        for file in files:
            out_files.append(os.path.join(file_dir, file))
    return out_files


def get_files(paths, target='.xml', with_recurrence=True):
    out_files = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if with_recurrence:
            files = list_dirs(path)
            for file in files:
                if os.path.splitext(file)[1] == target:
                    out_files.append(file)
        else:
            for file in os.scandir(path):
                if os.path.splitext(file)[1] == target:
                    out_files.append(file.path)
    return out_files


def safe_make_dirs(tdir):
    if not os.path.isdir(tdir):
        os.makedirs(tdir)


def get_img_label_pair(img_files, label_files, return_miss=False):
    out_pairs = []
    miss_ = []
    img_name_list = list(map(lambda s: os.path.splitext(os.path.split(s)[-1])[0], img_files))
    for label_file in label_files:
        img_name = os.path.splitext(os.path.split(label_file)[-1])[0]
        try:
            i = img_name_list.index(img_name)
            out_pairs.append([img_files[i], label_file])
        except ValueError as e:
            print("{} miss matched.".format(label_file))
            if return_miss:
                miss_.append(label_file)

    if return_miss:
        return out_pairs, miss_
    return out_pairs


def get_img_label_tri(img_files, json_files, png_files):
    out_pairs = []
    img_name_list = list(map(lambda s: os.path.splitext(os.path.split(s)[-1])[0], img_files))
    json_name_list = list(map(lambda s: os.path.splitext(os.path.split(s)[-1])[0], json_files))
    for png_file in png_files:
        img_name = os.path.splitext(os.path.split(png_file)[-1])[0]
        try:
            i = img_name_list.index(img_name)
            j = json_name_list.index(img_name)
            out_pairs.append([img_files[i], json_files[j], png_file])
        except ValueError as e:
            print("{} miss matched.".format(png_file))

    return out_pairs

def get_img_label_quadra(img_files, json_files, png_files):
    out_pairs = []
    img_name_list = list(map(lambda s: os.path.splitext(os.path.split(s)[-1])[0], img_files))
    seg_json_name_list = []
    kpt_json_name_list = []
    for json_file in json_files:
        if json_file.endswith('keypoints.json'):
            kpt_json_name_list.append(os.path.splitext(os.path.split(json_file)[-1])[0][:-10])
            seg_json_name_list.append(0)
        else:
            kpt_json_name_list.append(0)
            seg_json_name_list.append(os.path.splitext(os.path.split(json_file)[-1])[0])

    for png_file in png_files:
        img_name = os.path.splitext(os.path.split(png_file)[-1])[0]
        try:
            i = img_name_list.index(img_name)
            j = seg_json_name_list.index(img_name)
            k = kpt_json_name_list.index(img_name)
            out_pairs.append([img_files[i], json_files[j], json_file[k], png_file])
        except ValueError as e:
            print("{} miss matched.".format(png_file))

    return out_pairs

def is_duplicate(paths, target=".xml"):
    files = []
    for path in paths:
        files.extend(get_files(path, target=target))
    name_dict = {}
    for file in files:
        name = os.path.splitext(os.path.split(file)[-1])[0]
        if name in name_dict:
            print(file)
            print(name_dict[name])
        else:
            name_dict[name] = file
    print("name set length:{}, files length:{}".format(len(name_dict), len(files)))


def fill_area_multichn(img,chn, points):
    img_p = img.copy()
    for key_points in points:
        a = np.array([key_points])
        a = a.astype(np.int32)
        img_p = cv2.fillPoly(img_p[:,:,chn], a, color=1)
    return img_p

def cv_fill_area(img, points, color=None):
    if color is None:
        color = 0
    img_p = img.copy()
    for key_points in points:
        a = np.array([key_points])
        a = a.astype(np.int32)
        img_p = cv2.fillPoly(img_p, a, color=color)
    return img_p
