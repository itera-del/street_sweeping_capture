import cv2
import numpy as np
import random
import time
import os
from math import sqrt
import numpy as np


def get_time_str():
    # ex:'2018_03_30_20_34'
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime((time.time())))


def create_save_path(sp_name=None):
    time_stamp = get_time_str()
    if sp_name:
        time_stamp = sp_name + '_' + time_stamp
    save_ckpt_path = 'torch_model/{}/ckpt/'.format(time_stamp)
    save_model_path = 'torch_model/{}/model/'.format(time_stamp)
    if not (os.path.exists('torch_model/{}'.format(time_stamp))):
        os.makedirs('torch_model/{}'.format(time_stamp))
    if not (os.path.exists(save_ckpt_path)):
        os.makedirs(save_ckpt_path)
    if not (os.path.exists(save_model_path)):
        os.makedirs(save_model_path)
    return save_ckpt_path, save_model_path


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = s_rec1 + s_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def img_crop(img, annotation, rand_flag=True, rate_h=0.1, rate_w=0.1, bias_rate_h=0.1, bias_rate_w=0.2, crop_size_h=160,
             crop_size_w=224, inference_flag=False, resize_flag=True):
    pass
    # TODO
    # annotation:[xmin,ymin,width,height] (original number)
    # img: HWC image
    # RandFlag: do random crop or not
    # RateH, RateW : max enlarge rate (random)
    # CropSizeH, CropSizeW: Resize output image to a fix size

    H, W, _ = img.shape
    x = int(annotation[0])
    y = int(annotation[1])
    w = int(annotation[2])
    h = int(annotation[3])
    assert w > 0
    assert h > 0
    # print(H,W,x,y,w,h)
    if rand_flag:
        ran_w = 1 + (random.random() * rate_w + bias_rate_w)
        ran_h = 1 + (random.random() * rate_h + bias_rate_h)
        new_w = round(w * ran_w)
        new_h = round(h * ran_h)
        xx = random.randint(max(0, x - round(new_w - w) + round(w * bias_rate_w / 2)),
                            max(x - round(w * bias_rate_w / 2), 1))
        yy = random.randint(max(0, y - round(new_h - h) + round(h * bias_rate_h / 2)),
                            max(y - round(h * bias_rate_h / 2), 1))
    else:
        ran_w = 1 + bias_rate_w
        ran_h = 1 + bias_rate_h
        new_w = max(round(w * ran_w), 10)
        new_h = max(round(h * ran_h), 10)
        xx = max(round(x - (new_w - w) * 0.5), 0)
        yy = max(round(y - (new_h - h) * 0.5), 0)
    if yy + new_h >= H:
        if (H - new_h - 1) < 0:
            yy = 0
            new_h = H - 1
        else:
            yy = H - new_h - 1
    if xx + new_w >= W:
        if (W - new_w - 1) < 0:
            xx = 0
            new_w = W - 1
        else:
            xx = W - new_w - 1
    img_c = img[yy:yy + new_h, xx:xx + new_w, :]
    if resize_flag:
        img_re = cv2.resize(img_c, (crop_size_w, crop_size_h))
    else:
        img_re = img_c
    annotation_re = np.array(
        [float((x - xx) / (new_w / float(crop_size_w))), float((y - yy) / (new_h / float(crop_size_h))),
         w / (new_w / float(crop_size_w)), h / (new_h / float(crop_size_h))])
    if inference_flag:
        return img_re, np.array([xx, yy, new_w, new_h])
    return img_re, annotation_re


def modify_over_crop_box(annotation_re, crop_size_w, crop_size_h):
    x, y, w, h = annotation_re
    right, bottom = x + w, y + h
    x = max(x, -1)
    y = max(y, -1)
    right = min(crop_size_w + 1, right)
    bottom = min(crop_size_h + 1, bottom)
    return [x, y, right - x + 1, bottom - y + 1]


def img_crop_bbox(img, annotation, bbox, rand_flag=True, rate_h=0.1, rate_w=0.1, bias_rate_h=0.1, bias_rate_w=0.2,
                  crop_size_h=160, crop_size_w=224, log_file=None, inference_flag=False, resize_flag=True):
    pass
    # annotation:[car_annotation, plate_annotation]=[[xmin,ymin,width,height],[xmin,ymin,width,height]]  (original number)
    # img: HWC image
    # bbox [left, top, width, height]
    # RandFlag: do random crop or not
    # RateH, RateW : max enlarge rate (random)
    # CropSizeH, CropSizeW: Resize output image to a fix size
    log_flag = False if log_file is None else True
    H, W, _ = img.shape
    x, y, w, h = bbox
    assert w > 0
    assert h > 0
    if rand_flag:
        ran_w = 1 + (max(0.01, random.random() * rate_w) + bias_rate_w)
        ran_h = 1 + (max(0.01, random.random() * rate_h) + bias_rate_h)
        new_w = int(round(w * ran_w))
        new_h = int(round(h * ran_h))
        x_start = max(int(round(x - new_w + w + w * bias_rate_w / 2)), 0)
        x_end = max(int(round(x - w * bias_rate_w / 2)), 1)
        if x_start == x_end:
            xx = x_start
        else:
            xx = random.randint(x_start, x_end)
        y_start = int(max(0, int(y - (new_h - h) + int(h * bias_rate_h / 2) + 0.5)))
        y_end = max(int(round(y - int(h * bias_rate_h / 2))), 1)
        if y_start == y_end:
            yy = y_start
        else:
            assert y_start < y_end
            yy = random.randint(y_start, y_end)

        if log_flag:
            log_file.write(
                "new_w:{},new_h:{},xx:{},yy:{},x_start:{},x_end:{},y_start:{},y_end:{}\n".format(new_w, new_h, xx, yy,
                                                                                                 x_start, x_end,
                                                                                                 y_start, y_end))
    else:
        ran_w = 1 + bias_rate_w
        ran_h = 1 + bias_rate_h
        new_w = max(int(round(w * ran_w)), 10)
        new_h = max(int(round(h * ran_h)), 10)
        xx = max(int(round(x - (new_w - w) * 0.5)), 0)
        yy = max(int(round(y - (new_h - h) * 0.5)), 0)
    if yy + new_h >= H:
        if (H - new_h - 1) < 0:
            yy = 0
            new_h = H - 1
        else:
            yy = H - new_h - 1
    if xx + new_w >= W:
        if (W - new_w - 1) < 0:
            xx = 0
            new_w = W - 1
        else:
            xx = W - new_w - 1
    img_c = img[yy:yy + new_h, xx:xx + new_w, :]
    if resize_flag:
        try:
            img_re = cv2.resize(img_c, (crop_size_w, crop_size_h))
        except cv2.error:
            print(bbox)
            print(annotation)
            print([yy, yy + new_h, xx, xx + new_w])
            raise AssertionError
    else:
        img_re = img_c
    if inference_flag:
        return img_re, np.array([xx, yy, new_w, new_h])
    car_annotation = [float((annotation[0][0] - xx) / (new_w / float(crop_size_w))),
                      float((annotation[0][1] - yy) / (new_h / float(crop_size_h))),
                      annotation[0][2] / (new_w / float(crop_size_w)), annotation[0][3] / (new_h / float(crop_size_h))]
    car_annotation = modify_over_crop_box(car_annotation, crop_size_w, crop_size_h)
    plate_annotation = [float((annotation[1][0] - xx) / (new_w / float(crop_size_w))),
                        float((annotation[1][1] - yy) / (new_h / float(crop_size_h))),
                        annotation[1][2] / (new_w / float(crop_size_w)),
                        annotation[1][3] / (new_h / float(crop_size_h))]
    annotation_re = np.array([car_annotation, plate_annotation])
    return img_re, annotation_re


def check_bbox_type(annotation, bbox):
    x, y, w, h = bbox
    left, top, right, bottom = x, y, x + w, y + h
    anno_left, anno_top, anno_w, anno_h = annotation[0]
    anno_right, anno_bottom = anno_w + anno_left, anno_top + anno_h
    left_distance = abs(anno_left - left)
    right_distance = abs(anno_right - right)
    # print(anno_w, left_distance, right_distance)
    if (left_distance - right_distance) > (0.25 * anno_w):
        return "right"
    elif (right_distance - left_distance) > (0.25 * anno_w):
        return "left"
    elif abs(right_distance - left_distance) < (0.08 * anno_w):
        return "down"
    else:
        return None


def img_over_crop_bbox(img, annotation, bbox, rand_flag=True, rate_h=0.1, rate_w=0.1, bias_rate_h=0.1,
                       bias_rate_w=0.2, crop_size_h=160, crop_size_w=224, left_right_over_crop_rate=0.2,
                       up_down_over_crop_rate=0.2, mode=None, over_crop_rate=0.8, log_file=None):
    pass
    # annotation:[car_annotation, plate_annotation]=[[xmin,ymin,width,height],[xmin,ymin,width,height]]  (original number)
    # img: HWC image
    # bbox [x, y, width, height]
    # RandFlag: do random crop or not
    # RateH, RateW : max enlarge rate (random)
    # CropSizeH, CropSizeW: Resize output image to a fix size
    # left_right_over_crop_rate, up_down_over_crop_rate max over crop rate of left and right
    # mode in ["down", "left", "right"]

    log_flag = False if log_file is None else True
    H, W, _ = img.shape
    bbox = [int(bbox[i]) for i in range(len(bbox))]
    x, y, w, h = bbox
    if rand_flag:
        if (mode is None) and (random.random() > over_crop_rate):
            mode = check_bbox_type(annotation, bbox)
            # print(mode)
        # can not over crop, return normal crop result.
        if mode is None:
            return img_crop_bbox(img, annotation, bbox, rand_flag, rate_h, rate_w * 2, bias_rate_h, bias_rate_w,
                                 crop_size_h, crop_size_w, log_file=log_file)

        # print(H,W,x,y,w,h)
        ran_w_base = (random.random() * rate_w + bias_rate_w) / 2
        ran_h_base = (random.random() * rate_h + bias_rate_h) / 2
        if "left" in mode:
            x1_start = x
            x1_end = round(x + w * left_right_over_crop_rate)
            x2_start = min(x + w * (1 + bias_rate_w / 2), 1919)
            x2_end = min(x + w * (1 + ran_w_base), 1919)
        elif "right" in mode:
            x1_start = max(0, round(x - w * ran_w_base))
            x1_end = max(0, round(x - w * bias_rate_w / 2))
            x2_start = min(x + w * (1 - left_right_over_crop_rate), 1919)
            x2_end = min(x + w, 1919)
        else:
            x1_start = max(0, round(x - w * ran_w_base))
            x1_end = max(round(x - w * bias_rate_w / 2), 1)
            x2_start = min(x + w * (1 + bias_rate_w / 2), 1919)
            x2_end = min(x + w * (1 + ran_w_base), 1919)
        if "down" in mode:
            y1_start = max(0, y - ran_h_base * h)
            y1_end = max(0, y - h * bias_rate_h / 2)
            y2_start = min(1079, y + h * (1 - up_down_over_crop_rate))
            y2_end = min(1079, y + h)
        else:
            y1_start = max(0, y - ran_h_base * h)
            y1_end = max(0, y - h * bias_rate_h / 2)
            y2_start = min(1079, y + h * (1 + bias_rate_h / 2))
            y2_end = min(1079, y + h * (1 + ran_h_base))
        x1_start, x1_end = int(x1_start), int(x1_end)
        x2_start, x2_end = int(x2_start), int(x2_end)
        y1_start, y1_end = int(y1_start), int(y1_end)
        y2_start, y2_end = int(y2_start), int(y2_end)
        assert x1_start <= x1_end
        assert x2_start <= x2_end
        assert y1_start <= y1_end
        assert y2_start <= y2_end
        # print("x1_start:{},x1_end:{}".format(x1_start, x1_end))
        # print("x2_start:{},x2_end:{}".format(x2_start, x2_end))
        # print("y1_start:{},y1_end:{}".format(y1_start, y1_end))
        # print("y2_start:{},y2_end:{}".format(y2_start, y2_end))
        xx = random.randint(x1_start, x1_end)
        new_w = random.randint(x2_start, x2_end) - xx
        yy = random.randint(y1_start, y1_end)
        new_h = random.randint(y2_start, y2_end) - yy
        if log_flag:
            log_file.write(
                "new_w:{},new_h:{},xx:{},yy:{},x_start:{},x_end:{},y_start:{},y_end:{},mode:{}\n".format(new_w, new_h,
                                                                                                         xx, yy,
                                                                                                         x1_start,
                                                                                                         x1_end,
                                                                                                         y1_start,
                                                                                                         y1_end,
                                                                                                         mode))
    else:
        ran_w = 1 + bias_rate_w
        ran_h = 1 + bias_rate_h
        new_w = max(round(w * ran_w), 10)
        new_h = max(round(h * ran_h), 10)
        xx = max(round(x - (new_w - w) * 0.5), 0)
        yy = max(round(y - (new_h - h) * 0.5), 0)
    if yy + new_h >= H:
        if (H - new_h - 1) < 0:
            yy = 0
            new_h = H - 1
        else:
            yy = H - new_h - 1
    if xx + new_w >= W:
        if (W - new_w - 1) < 0:
            xx = 0
            new_w = W - 1
        else:
            xx = W - new_w - 1
    assert new_w > 20
    assert new_h > 20
    img_c = img[yy:yy + new_h, xx:xx + new_w, :]
    # img_re = cv2.resize(img_c, (crop_size_w, crop_size_h))
    try:
        img_re = cv2.resize(img_c, (crop_size_w, crop_size_h))
    except cv2.error:
        print(bbox)
        print(annotation)
        print([yy, yy + new_h, xx, xx + new_w])
        raise AssertionError
    car_annotation = [float((annotation[0][0] - xx) / (new_w / float(crop_size_w))),
                      float((annotation[0][1] - yy) / (new_h / float(crop_size_h))),
                      annotation[0][2] / (new_w / float(crop_size_w)), annotation[0][3] / (new_h / float(crop_size_h))]
    car_annotation = modify_over_crop_box(car_annotation, crop_size_w, crop_size_h)
    plate_annotation = [float((annotation[1][0] - xx) / (new_w / float(crop_size_w))),
                        float((annotation[1][1] - yy) / (new_h / float(crop_size_h))),
                        annotation[1][2] / (new_w / float(crop_size_w)),
                        annotation[1][3] / (new_h / float(crop_size_h))]
    annotation_re = np.array([car_annotation, plate_annotation])
    return img_re, annotation_re


def img_flip(img, annotation):
    pass
    H, W, _ = img.shape
    img_flp = cv2.flip(img, 1)
    car_annotation_flip = [W - (annotation[0][0] + annotation[0][2]) + 1, annotation[0][1], annotation[0][2],
                           annotation[0][3]]
    plate_annotation_flip = [W - (annotation[1][0] + annotation[1][2]) + 1, annotation[1][1], annotation[1][2],
                             annotation[1][3]]
    annotation_flip = np.array([car_annotation_flip, plate_annotation_flip])
    return img_flp, annotation_flip


def brightness_change(img, value=0):
    if value > 0:
        shadow = value
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + value
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow
    img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    return img


def saturation_change(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >= 0:
        lim = 255 - value
        s[s > lim] = 255
        s[s <= lim] += value
    else:
        value = abs(value)
        s[s < value] = 0
        s[s >= value] -= value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def contrast_change(img, contrast=0):
    buf = img.copy()
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def add_noise(img, noise_typ="random"):
    if noise_typ == "random":
        noise_typ = random.choice(['gauss', 's&p', 'poisson', 'speckle'])
    # print(noise_typ)
    if noise_typ == "gauss":
        H, W, C = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (H, W, C))
        gauss = gauss.reshape(H, W, C)
        noisy = img + gauss
        return noisy.astype(np.uint8)
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy.astype(np.uint8)
    elif noise_typ == "speckle":
        H, W, C = img.shape
        gauss = np.random.randn(H, W, C) * (np.random.random() + 0.1) * 0.15
        gauss = gauss.reshape(H, W, C)
        noisy = img + img * gauss
        return noisy.astype(np.uint8)


def mosaic(img, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param img: opencv frame
    :param int x : 马赛克左顶点
    :param int y: 马赛克右顶点
    :param int w: 马赛克宽
    :param int h: 马赛克高
    :param int neighbor: 马赛克每一块的宽
    """
    fh, fw = img.shape[0], img.shape[1]
    if (y > fh) or (x > fw):
        return img
    w = min(w, fw - x - 1)
    h = min(h, fh - y - 1)
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neighbor 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = img[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(img, left_up, right_down, color, -1)
    return img


def add_mosaic(img, annotation, neighbor=5):
    x, y, w, h = annotation
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    left_top_x = x + int((0.35 * random.random() + 0.1) * w)
    left_top_y = y + int((0.35 * random.random() + 0.1) * h)
    right_bottom_x = x + int((0.35 * random.random() + 0.55) * w)
    right_bottom_y = y + int((0.35 * random.random() + 0.8) * h)
    # print(left_top_x, left_top_y, right_bottom_x, right_bottom_y)
    img = mosaic(img, left_top_x, left_top_y, right_bottom_x - left_top_x, right_bottom_y - left_top_y, neighbor)

    return img


def random_erasing(img, bbox, p=0.5, r=None, s=None):
    if random.random() < p:
        return img
    else:
        if r is None:
            r = [0.3, 10 / 3]
        if s is None:
            s = [0.02, 0.2]
        img_h, img_w, _ = img.shape
        x, y, w, h = [round(b) for b in bbox]
        if (x < img_w) or (y < img_h):
            return img
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        box_s = h * w
        while True:
            re = random.random() * (r[1] - r[0]) + r[0]
            se = (random.random() * (s[1] - s[0]) + s[0]) * box_s
            he = int(sqrt(se * re))
            we = int(sqrt(se / re))
            xe = random.randint(x, x + w)
            ye = random.randint(y, y + h)
            if xe + we <= y + w and ye + he <= y + h:
                img[ye:ye + he, xe:xe + we, :] = np.random.randint(0, 255, size=(he, we, 3), dtype=np.uint8)
                return img


def add_erasing(img, annotation, p=0.5, r=None, s=None):
    img = random_erasing(img, annotation, p, r, s)
    if random.random() < p:
        img = random_erasing(img, [0, 0, img.shape[1], img.shape[0]], p=1)
    return img


def data_enhance(img_o, annotation, enhance_list):
    # enhance_list in ["flip", "brightness", "saturation", "contrast", "noise"]
    img = img_o.copy()
    # print(enhance_list)
    if "flip" in enhance_list:
        img, annotation = img_flip(img, annotation)
    if "brightness" in enhance_list:
        img = brightness_change(img, random.randint(-70, 70))
    if "saturation" in enhance_list:
        img = saturation_change(img, random.randint(-30, 30))
    if "contrast" in enhance_list:
        img = contrast_change(img, random.randint(-30, 30))
    if "noise" in enhance_list:
        img = add_noise(img)
    if "add_patch" in enhance_list:
        img = add_erasing(img, annotation[0])
    if "add_mosaic" in enhance_list:
        img = add_mosaic(img, annotation[0])
    return img, annotation


def generate_random_processing_img(train_img_name, train_annotation, train_bbox,
                                   img_path="../data/MTCNN_VOC/JPEGImages/", rand_flag=True):
    """
    doing random processing image
    :param train_img_name:  (str or ndarray): Image file path or image data array
    :param train_annotation:  (list): bounding box of annotation, [[left, top, right, bottom], [...] , ...]
    :param train_bbox:  (list): bounding box of object, [[left, top, right, bottom], [...] , ...]
    :param img_path: (str): data path.
    :param rand_flag: (bool, optional): Determines whether apply random processing to image; Default **True**
    :return: img_c, annotation_c : (list): data after processing
    """
    enhance_list = ["flip", "brightness", "saturation", "contrast", "noise", "add_patch", "add_mosaic"]
    if type(train_img_name) == str or type(train_img_name) == np.str_:
        img = cv2.imread(img_path + train_img_name.strip().split('/')[-1])
    else:
        img = train_img_name
    rate_h = 0.2
    rate_w = 0.3 if rand_flag else 0.2
    anno_left, anno_top, anno_right, anno_bottom = train_annotation[0]
    annotation = [[anno_left, anno_top, anno_right - anno_left, anno_bottom - anno_top]]
    anno_left, anno_top, anno_right, anno_bottom = train_annotation[1]
    annotation.append([anno_left, anno_top, anno_right - anno_left, anno_bottom - anno_top])
    left, top, right, bottom = train_bbox
    bbox = [left, top, right - left, bottom - top]
    if rand_flag:
        img_c, annotation_c = img_over_crop_bbox(img, annotation, bbox, rand_flag=True, rate_h=rate_h,
                                                 rate_w=rate_w / 2,
                                                 left_right_over_crop_rate=0.2, bias_rate_h=0.1, bias_rate_w=0.1,
                                                 up_down_over_crop_rate=0.1, mode=None)
        if random.random() > 0.5:
            process_list = random.sample(enhance_list, k=random.randint(1, len(enhance_list)))
            img_c, annotation_c = data_enhance(img_c, annotation_c, process_list)
    else:
        img_c, annotation_c = img_over_crop_bbox(img, annotation, bbox, rand_flag=rand_flag, rate_h=rate_h,
                                                 rate_w=rate_w, bias_rate_h=0.1, bias_rate_w=0.1, mode=None)
    return img_c, annotation_c


def generate_random_processing_img_v2(train_image, train_annotation, train_bbox, rand_flag=True):
    enhance_list = ["flip", "brightness", "saturation", "contrast", "noise"]
    img = train_image
    rate_h = 0.2
    rate_w = 0.3 if rand_flag else 0.2
    anno_left, anno_top, anno_right, anno_bottom = train_annotation[0]
    annotation = [[anno_left, anno_top, anno_right - anno_left, anno_bottom - anno_top]]
    anno_left, anno_top, anno_right, anno_bottom = train_annotation[1]
    annotation.append([anno_left, anno_top, anno_right - anno_left, anno_bottom - anno_top])
    left, top, right, bottom = train_bbox
    bbox = [left, top, right - left, bottom - top]
    if rand_flag:
        img_c, annotation_c = img_over_crop_bbox(img, annotation, bbox, rand_flag=True, rate_h=rate_h,
                                                 rate_w=rate_w / 2,
                                                 left_right_over_crop_rate=0.3, bias_rate_h=0.1, bias_rate_w=0.1,
                                                 up_down_over_crop_rate=0.3, mode=None)
        if random.random() > 0.5:
            process_list = random.sample(enhance_list, k=random.randint(1, len(enhance_list)))
            img_c, annotation_c = data_enhance(img_c, annotation_c, process_list)
    else:
        img_c, annotation_c = img_over_crop_bbox(img, annotation, bbox, rand_flag=rand_flag, rate_h=rate_h,
                                                 rate_w=rate_w, bias_rate_h=0.1, bias_rate_w=0.1, mode=None)
    return img_c, annotation_c


def cv_plot_rectangle(img, bbox, color=None, mode='xywh', thickness=3):
    if color is None:
        color = (255, 0, 0)
    if mode == 'xywh':
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, w + x, h + y
    elif mode == 'ltrb':
        xmin, ymin, xmax, ymax = bbox
    else:
        print("Unknown plot mode")
        return None
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    img_p = img.copy()
    return cv2.rectangle(img_p, (xmin, ymin),
                         (xmax, ymax), color=color, thickness=thickness)


def draw_detection_result(img, bboxes, mode='xywh', color=None):
    if color is None:
        color = (255, 0, 0)
    for key, values in bboxes.items():
        for box in values:
            if len(box) == 5:
                if isinstance(box[0], float):
                    box = [int(b + 0.5) for b in box[:4]]
                # img = cv2.putText(img, f"{key}:" + "%.3f" % box[-1], (box[0], box[1]),
                #                   cv2.FONT_HERSHEY_COMPLEX,
                #                   1, color, 2)
                img = cv_plot_rectangle(img, box[:4], mode=mode, color=color)
            else:
                if isinstance(box[0], float):
                    box = [int(b + 0.5) for b in box]
                # img = cv2.putText(img, f"{key}", (box[0], box[1] + 10), cv2.FONT_HERSHEY_COMPLEX,
                #                   1, color, 2)
                img = cv_plot_rectangle(img, box, mode=mode, color=color)
    return img


def safe_make_dirs(tdir):
    if not os.path.isdir(tdir):
        os.makedirs(tdir)


def clear_path(dir):
    if os.path.exists(dir):
        files = os.listdir(dir)
        for file in files:
            os.remove(os.path.join(dir, file))


def convert_tensor_2_cvimg(img_tensor, cfg):
    npimg = img_tensor.to('cpu').numpy()
    img = np.transpose(((npimg[0] * cfg.img_scale) + cfg.img_mean), (1, 2, 0)).astype(np.uint8)
    return img


def bbox_iou(bbox1, bbox2):
    """calculate iou for given bounding box
    :param bbox1: (list) x y w h
    :param bbox2: (list) x y w h
    :return: iou (float)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    min_x = min(x1 + w1, x2 + w2)
    min_y = min(y1 + h1, y2 + h2)
    inter_area = max(0, min_x - max_x) * max(0, min_y - max_y)
    return inter_area / (w1 * h1 + w2 * h2 - inter_area)


def distance_iou_loss(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with corresponding boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    # b1_len = boxes1.size(0)
    # b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    # print(boxes1.device, boxes2.device)

    dx = (b1x2.min(b2x2) - b1x1.max(b2x1)).clamp(min=0)
    dy = (b1y2.min(b2y2) - b1y1.max(b2y1)).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2) - intersections

    iou = intersections / unions

    p_distance = ((boxes1[:, :2] - boxes2[:, :2]) ** 2).sum(dim=1, keepdim=True)
    c_distance = ((b1x2.max(b2x2) - b1x1.min(b2x1)) ** 2 + (b1y2.max(b2y2) - b1y1.min(b2y1)) ** 2).clamp(min=0.1)
    d_iou_loss = 1 - iou + p_distance / c_distance
    return d_iou_loss
