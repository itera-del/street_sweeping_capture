from tkinter import W
import cv2
import io
import json
import numpy as np
import sys

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from infer.lpr_seg import LPRSegCaffe, LPRSeg2HeadCaffe, LPRSegColorClassCaffe
# # ocr
# from infer.lpr import LPRCaffe, ocr_labels_zd

# paddle_ocr
from infer.lpr import ocr_labels_zd
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.paddle.infer.lpr import LPRCaffe

class LPRSegOcrcaffe(object):

    def __init__(self, lpr_seg_prototxt, lpr_seg_model_path, lpr_ocr_prototxt, lpr_ocr_model_path, lpr_seg_input_shape=(128, 64), lpr_ocr_input_shape=(256, 64), lpr_ocr_ocr_labels=ocr_labels_zd, lpr_ocr_prefix_beam_search_bool=False, gpu_bool=False, dict_path="/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/inference/zd_dict.txt"):
        
        self.lpr_seg_prototxt = lpr_seg_prototxt
        self.lpr_seg_model_path = lpr_seg_model_path
        self.lpr_seg_input_shape = lpr_seg_input_shape

        self.lpr_ocr_prototxt = lpr_ocr_prototxt
        self.lpr_ocr_model_path = lpr_ocr_model_path
        self.lpr_ocr_input_shape = lpr_ocr_input_shape
        self.lpr_ocr_ocr_labels = lpr_ocr_ocr_labels
        self.lpr_ocr_prefix_beam_search_bool = lpr_ocr_prefix_beam_search_bool
        self.dict_path = dict_path
        self.gpu_bool = gpu_bool

        # 白天 1080p
        self.ignore_kind_min_size_threh = 12
        self.ignore_num_min_size_threh = 20
        # # 白天
        # self.ignore_kind_min_size_threh = 13        # 12 -> 13 - 14
        # self.ignore_num_min_size_threh = 23         # 20 -> 22 - 23
        # # 夜间
        # self.ignore_kind_min_size_threh = 18
        # self.ignore_num_min_size_threh = 28
        
        self.hisi_min_size_threh = 32

        # seg
        # self.lpr_seg = LPRSegCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)
        # self.lpr_seg = LPRSeg2HeadCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "dataset_zd_dict", "dataset_zd_dict_color", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)
        # self.lpr_seg = LPRSegColorClassCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city_2022", "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)
        # self.lpr_seg = LPRSegColorClassCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city", "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)
        self.lpr_seg = LPRSegColorClassCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "script.lpr.dataset.dataset_zd_shate.dataset_dict.dataset_zd_dict_city", "script.lpr.dataset.dataset_zd_shate.dataset_dict.dataset_zd_dict_color", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)

        # # # ocr
        # self.lpr_ocr = LPRCaffe(self.lpr_ocr_prototxt, self.lpr_ocr_model_path, self.lpr_ocr_input_shape, self.lpr_ocr_ocr_labels, self.lpr_ocr_prefix_beam_search_bool, gpu_bool=self.gpu_bool)
        # paddle_ocr
        self.lpr_ocr = LPRCaffe(self.lpr_ocr_model_path, self.lpr_ocr_prototxt, self.dict_path, gpu_bool=self.gpu_bool)

        self.interval_img_path = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.jpg"
        self.interval_json_path = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.json"
        self.interval_img = self.load_interval_img()
        # cv2.imwrite("/mnt/huanyuan/model/image/lpr/zd/img/interval_img.jpg", self.interval_img)


    def load_interval_img(self):

        # interval
        interval_img = np.zeros((0, 0))
        ori_interval_img = cv2.imread(self.interval_img_path)
        with io.open(self.interval_json_path, "r") as f:
            data = json.load(f)
            f.close()
        pts = np.array(data['shapes'][0]["points"], np.int32).reshape((-1, 1, 2))
        x1 = np.min(pts[:, 0, 0])
        x2 = np.max(pts[:, 0, 0])
        y1 = np.min(pts[:, 0, 1])
        y2 = np.max(pts[:, 0, 1])
        interval_img = ori_interval_img[y1:y2, x1:x2]
        
        return interval_img


    def run(self, img):

        # init
        bool_ignore_kind = False
        bool_ignore_num = False
        concate_img = np.zeros((0, 0))

        seg_bbox, seg_info = self.lpr_seg.run(img)

        # kind
        kind_img = np.zeros((0, 0))
        if 'kind' in seg_bbox:
            kind_box = seg_bbox['kind'][0]

            x1, x2 = kind_box[0], kind_box[0] + kind_box[2]
            y1, y2 = kind_box[1], kind_box[1] + kind_box[3]
            h = y2 - y1
            w = x2 - x1

            # 往外扩张
            x1 = max(0, int(x1 - 0.08 * w))
            x2 = min(img.shape[1], int(x2 + 0.08 * w))
            y1 = max(0, int(y1 - 0.08 * h))
            y2 = min(img.shape[0], int(y2 + 0.08 * h))
            h = y2 - y1
            w = x2 - x1

            if w < self.ignore_kind_min_size_threh or h < self.ignore_kind_min_size_threh:
                bool_ignore_kind = True
            
            # 适配海思开发板，最小 roi 大小
            if w < self.hisi_min_size_threh:
                x1 = max(0, int(((x1 + x2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                x2 = min(img.shape[1], x1 + self.hisi_min_size_threh)
                w = self.hisi_min_size_threh                

            if h < self.hisi_min_size_threh:
                y1 = max(0, int(((y1 + y2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                y2 = min(img.shape[0], y1 + self.hisi_min_size_threh)
                h = self.hisi_min_size_threh 

            seg_bbox['kind'][0][0] = x1
            seg_bbox['kind'][0][1] = y1
            seg_bbox['kind'][0][2] = w
            seg_bbox['kind'][0][3] = h

            kind_img = img[y1:y2, x1:x2]

        # num 
        num_img = np.zeros((0, 0))
        if 'num' in seg_bbox:
            num_box = seg_bbox['num'][0]

            x1, x2 = num_box[0], num_box[0]+num_box[2]
            y1, y2 = num_box[1], num_box[1]+num_box[3]
            h = y2 - y1
            w = x2 - x1

            # 往外扩张
            x1 = max(0, int(x1 - 0.08 * w))
            x2 = min(img.shape[1], int(x2 + 0.08 * w))
            y1 = max(0, int(y1 - 0.08 * h))
            y2 = min(img.shape[0], int(y2 + 0.08 * h))
            h = y2 - y1
            w = x2 - x1

            if w < self.ignore_num_min_size_threh or h < self.ignore_num_min_size_threh:
                bool_ignore_num = True

            # 适配海思开发板，最小 roi 大小
            if w < self.hisi_min_size_threh:
                x1 = max(0, int(((x1 + x2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                x2 = min(img.shape[1], x1 + self.hisi_min_size_threh)
                w = self.hisi_min_size_threh 

            if h < self.hisi_min_size_threh:
                y1 = max(0, int(((y1 + y2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                y2 = min(img.shape[0], y1 + self.hisi_min_size_threh)
                h = self.hisi_min_size_threh

            seg_bbox['num'][0][0] = x1
            seg_bbox['num'][0][1] = y1
            seg_bbox['num'][0][2] = w
            seg_bbox['num'][0][3] = h

            num_img = img[y1:y2, x1:x2]

        if 'kind' in seg_bbox or 'num' in seg_bbox:
            # resize
            kind_shape, num_shape, interval_shape = get_resize_shape(kind_img, num_img, self.interval_img)
            if 'kind' in seg_bbox:
                kind_img = cv2.resize(kind_img, kind_shape)
            if 'num' in seg_bbox:
                num_img = cv2.resize(num_img, num_shape)
            interval_img_copy = cv2.resize(self.interval_img, interval_shape)

            # concate
            if 'kind' in seg_bbox and 'num' in seg_bbox:
                concate_img = cv2.hconcat([kind_img, interval_img_copy, num_img])     
            elif 'kind' in seg_bbox:
                concate_img = cv2.hconcat([kind_img, interval_img_copy]) 
            elif 'num' in seg_bbox:
                concate_img = cv2.hconcat([interval_img_copy, num_img]) 

            # # ocr
            # gray_concate_img = cv2.cvtColor(concate_img, cv2.COLOR_BGR2GRAY)
            # ocr, ocr_score = self.lpr_ocr.run(gray_concate_img)
            # paddle_ocr
            ocr, ocr_score = self.lpr_ocr.run(concate_img)
        else:
            ocr = ''
            ocr_score = []

        ocr_ignore = bool_ignore_kind or bool_ignore_num
         
        return seg_bbox, seg_info, ocr, ocr_score, ocr_ignore, concate_img
    

class LPRSegOcrMoroccocaffe(object):

    def __init__(self, lpr_seg_prototxt, lpr_seg_model_path, lpr_ocr_prototxt, lpr_ocr_model_path, lpr_seg_input_shape=(128, 64), lpr_ocr_input_shape=(256, 64), lpr_ocr_ocr_labels=ocr_labels_zd, lpr_ocr_prefix_beam_search_bool=False, gpu_bool=False, dict_path="/yuanhuan/model/image/lpr/paddle_dict/plate_morocco_mask_20250515_NoAug/morocco_dict.txt"):
        
        self.lpr_seg_prototxt = lpr_seg_prototxt
        self.lpr_seg_model_path = lpr_seg_model_path
        self.lpr_seg_input_shape = lpr_seg_input_shape

        self.lpr_ocr_prototxt = lpr_ocr_prototxt
        self.lpr_ocr_model_path = lpr_ocr_model_path
        self.lpr_ocr_input_shape = lpr_ocr_input_shape
        self.lpr_ocr_ocr_labels = lpr_ocr_ocr_labels
        self.lpr_ocr_prefix_beam_search_bool = lpr_ocr_prefix_beam_search_bool
        self.dict_path = dict_path
        self.gpu_bool = gpu_bool

        # 白天
        self.ignore_kind_min_size_threh = 12
        self.ignore_city_min_size_threh = 12
        self.ignore_num_min_size_threh = 20
        
        self.hisi_min_size_threh = 32

        # seg
        self.lpr_seg = LPRSegCaffe(self.lpr_seg_prototxt, self.lpr_seg_model_path, "script.lpr.dataset.dataset_morocco.dataset_dict.dataset_morocco_dict_city", self.lpr_seg_input_shape, gpu_bool=self.gpu_bool)

        # # # ocr
        # self.lpr_ocr = LPRCaffe(self.lpr_ocr_prototxt, self.lpr_ocr_model_path, self.lpr_ocr_input_shape, self.lpr_ocr_ocr_labels, self.lpr_ocr_prefix_beam_search_bool, gpu_bool=self.gpu_bool)
        # paddle_ocr
        self.lpr_ocr = LPRCaffe(self.lpr_ocr_model_path, self.lpr_ocr_prototxt, self.dict_path, gpu_bool=self.gpu_bool)


    def run(self, img):

        # init
        bool_ignore_kind = False
        bool_ignore_num = False
        bool_ignore_city = False
        concate_img = np.zeros((0, 0))

        seg_bbox, seg_info = self.lpr_seg.run(img)

        # kind
        kind_img = np.zeros((0, 0))
        if 'kind' in seg_bbox:
            kind_box = seg_bbox['kind'][0]

            x1, x2 = kind_box[0], kind_box[0] + kind_box[2]
            y1, y2 = kind_box[1], kind_box[1] + kind_box[3]
            h = y2 - y1
            w = x2 - x1

            # 往外扩张
            x1 = max(0, int(x1 - 0.08 * w))
            x2 = min(img.shape[1], int(x2 + 0.08 * w))
            y1 = max(0, int(y1 - 0.08 * h))
            y2 = min(img.shape[0], int(y2 + 0.08 * h))
            h = y2 - y1
            w = x2 - x1

            if w < self.ignore_kind_min_size_threh or h < self.ignore_kind_min_size_threh:
                bool_ignore_kind = True
            
            # 适配海思开发板，最小 roi 大小
            if w < self.hisi_min_size_threh:
                x1 = max(0, int(((x1 + x2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                x2 = min(img.shape[1], x1 + self.hisi_min_size_threh)
                w = self.hisi_min_size_threh                

            if h < self.hisi_min_size_threh:
                y1 = max(0, int(((y1 + y2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                y2 = min(img.shape[0], y1 + self.hisi_min_size_threh)
                h = self.hisi_min_size_threh 

            seg_bbox['kind'][0][0] = x1
            seg_bbox['kind'][0][1] = y1
            seg_bbox['kind'][0][2] = w
            seg_bbox['kind'][0][3] = h

            kind_img = img[y1:y2, x1:x2]

        # num 
        num_img = np.zeros((0, 0))
        if 'num' in seg_bbox:
            num_box = seg_bbox['num'][0]

            x1, x2 = num_box[0], num_box[0]+num_box[2]
            y1, y2 = num_box[1], num_box[1]+num_box[3]
            h = y2 - y1
            w = x2 - x1

            # 往外扩张
            x1 = max(0, int(x1 - 0.08 * w))
            x2 = min(img.shape[1], int(x2 + 0.08 * w))
            y1 = max(0, int(y1 - 0.08 * h))
            y2 = min(img.shape[0], int(y2 + 0.08 * h))
            h = y2 - y1
            w = x2 - x1

            if w < self.ignore_num_min_size_threh or h < self.ignore_num_min_size_threh:
                bool_ignore_num = True

            # 适配海思开发板，最小 roi 大小
            if w < self.hisi_min_size_threh:
                x1 = max(0, int(((x1 + x2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                x2 = min(img.shape[1], x1 + self.hisi_min_size_threh)
                w = self.hisi_min_size_threh 

            if h < self.hisi_min_size_threh:
                y1 = max(0, int(((y1 + y2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                y2 = min(img.shape[0], y1 + self.hisi_min_size_threh)
                h = self.hisi_min_size_threh

            seg_bbox['num'][0][0] = x1
            seg_bbox['num'][0][1] = y1
            seg_bbox['num'][0][2] = w
            seg_bbox['num'][0][3] = h

            num_img = img[y1:y2, x1:x2]

        # city 
        city_img = np.zeros((0, 0))
        if 'city' in seg_bbox:
            city_box = seg_bbox['city'][0]

            x1, x2 = city_box[0], city_box[0]+city_box[2]
            y1, y2 = city_box[1], city_box[1]+city_box[3]
            h = y2 - y1
            w = x2 - x1

            # 往外扩张
            x1 = max(0, int(x1 - 0.08 * w))
            x2 = min(img.shape[1], int(x2 + 0.08 * w))
            y1 = max(0, int(y1 - 0.08 * h))
            y2 = min(img.shape[0], int(y2 + 0.08 * h))
            h = y2 - y1
            w = x2 - x1

            if w < self.ignore_city_min_size_threh or h < self.ignore_city_min_size_threh:
                bool_ignore_city = True

            # 适配海思开发板，最小 roi 大小
            if w < self.hisi_min_size_threh:
                x1 = max(0, int(((x1 + x2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                x2 = min(img.shape[1], x1 + self.hisi_min_size_threh)
                w = self.hisi_min_size_threh 

            if h < self.hisi_min_size_threh:
                y1 = max(0, int(((y1 + y2) / 2) - (self.hisi_min_size_threh  / 2) + 0.5))
                y2 = min(img.shape[0], y1 + self.hisi_min_size_threh)
                h = self.hisi_min_size_threh

            seg_bbox['city'][0][0] = x1
            seg_bbox['city'][0][1] = y1
            seg_bbox['city'][0][2] = w
            seg_bbox['city'][0][3] = h

            city_img = img[y1:y2, x1:x2]

        if 'kind' in seg_bbox or 'num' in seg_bbox or 'city' in seg_bbox:
            # resize
            kind_shape, num_shape, city_shape = get_resize_shape_w_city(kind_img, num_img, city_img)
            if 'kind' in seg_bbox:
                kind_img = cv2.resize(kind_img, kind_shape)
            if 'num' in seg_bbox:
                num_img = cv2.resize(num_img, num_shape)
            if 'city' in seg_bbox:
                city_img = cv2.resize(city_img, city_shape)

            # concate
            if 'kind' in seg_bbox and 'num' in seg_bbox and 'city' in seg_bbox:
                concate_img = cv2.hconcat([kind_img, city_img, num_img])     
            elif 'city' in seg_bbox and 'num' in seg_bbox: 
                concate_img = cv2.hconcat([city_img, num_img]) 
            elif 'num' in seg_bbox:
                concate_img = cv2.hconcat([num_img]) 

            if concate_img.shape == (0, 0):
                ocr = ''
                ocr_score = []
            else:

                # # ocr
                # gray_concate_img = cv2.cvtColor(concate_img, cv2.COLOR_BGR2GRAY)
                # ocr, ocr_score = self.lpr_ocr.run(gray_concate_img)
                # paddle_ocr
                ocr, ocr_score = self.lpr_ocr.run(concate_img)
                # from datetime import datetime
                # cv2.imwrite(f"/yuanhuan/data/temp_result/POLICE_MOROCCO_C27/2M5M_白天_0329/test/concate_img_{ocr}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", concate_img)
        else:
            ocr = ''
            ocr_score = []

        ocr_ignore = bool_ignore_kind or bool_ignore_num or bool_ignore_city
         
        return seg_bbox, seg_info, ocr, ocr_score, ocr_ignore, concate_img


def get_resize_shape(kind_img, num_img, interval_img):
    
    # init
    aligned_height = max(kind_img.shape[0], num_img.shape[0])
    
    # rate
    kind_rate = aligned_height / (kind_img.shape[0] + 1e-5)
    num_rate = aligned_height / (num_img.shape[0] + 1e-5)
    interval_rate = aligned_height / (interval_img.shape[0] + 1e-5)

    kind_shape = []
    kind_shape.append(int(kind_img.shape[1] * kind_rate))
    kind_shape.append(aligned_height)
    
    num_shape = []
    num_shape.append(int(num_img.shape[1] * num_rate))
    num_shape.append(aligned_height)
    
    interval_shape = []
    interval_shape.append(int(interval_img.shape[1] * interval_rate))
    interval_shape.append(aligned_height)
    
    kind_shape = tuple(kind_shape)
    num_shape = tuple(num_shape)
    interval_shape = tuple(interval_shape)
    return kind_shape, num_shape, interval_shape

def get_resize_shape_w_city(kind_img, num_img, city_img):

    # init
    aligned_height = max(kind_img.shape[0], num_img.shape[0], city_img.shape[0])
    
    # rate
    kind_rate = aligned_height / (kind_img.shape[0] + 1e-5)
    num_rate = aligned_height / (num_img.shape[0] + 1e-5)
    city_rate = aligned_height / (city_img.shape[0] + 1e-5)

    kind_shape = []
    kind_shape.append(int(kind_img.shape[1] * kind_rate))
    kind_shape.append(aligned_height)
    
    num_shape = []
    num_shape.append(int(num_img.shape[1] * num_rate))
    num_shape.append(aligned_height)
    
    city_shape = []
    city_shape.append(int(city_img.shape[1] * city_rate))
    city_shape.append(aligned_height)

    return kind_shape, num_shape, city_shape