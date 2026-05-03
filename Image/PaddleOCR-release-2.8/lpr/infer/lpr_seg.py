import cv2
from collections import OrderedDict
import importlib
import numpy as np
import sys
# import torch
# import torch.nn.functional as F

# # caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
# sys.path.insert(0, caffe_root + 'python')
import caffe

# # sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
# sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
# import models.NovaLaneNet as NovaLaneNet

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')


class LPRSegCaffe(object):
    
    def __init__(self, prototxt, model_path, dict_name, input_shape=(128, 64), gpu_bool=False):
        
        self.prototxt = prototxt
        self.model_path = model_path
        self.input_shape = input_shape
        self.gpu_bool = gpu_bool

        self.dict = importlib.import_module(dict_name) 
        
        self.model_init()

    
    def model_init(self):

        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)


    def model_forward(self, img):
        self.net.blobs['data'].data[...] = img
        preds = self.net.forward()["prob"]
        preds = np.squeeze(preds)
        preds = preds * (preds > 0.5)
        preds_mask = np.argmax(preds, axis=0).astype(np.uint8)
        return preds_mask


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask = self.model_forward(img)

        # mask
        seg_mask = cv2.resize(preds_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}
        for label_idx in range(len(self.dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.dict, seg_mask.copy(), self.dict.class_seg_label_group[label_idx], self.dict.class_seg_label_group_2_id_map[self.dict.class_seg_label_group[label_idx]], self.dict.class_seg_label_group_threh_map[self.dict.class_seg_label_group[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


class LPRSegOnnx(object):
    
    def __init__(self, model_path, dict_name, input_shape=(128, 64), gpu_bool=False):
        
        self.model_path = model_path
        self.input_shape = input_shape
        self.gpu_bool = gpu_bool

        self.dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + dict_name) 
        
        self.model_init()

    
    def model_init(self):

        import onnxruntime as ort
        self.net = ort.InferenceSession(self.model_path)


    def model_forward(self, img):

        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        input_dict = {}
        input_dict[self.net.get_inputs()[0].name] = img
        preds = self.net.run(None, input_dict)[0]

        preds = np.squeeze(preds)
        preds = preds * (preds > 0.5)
        preds_mask = np.argmax(preds, axis=0).astype(np.uint8)
        return preds_mask


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask = self.model_forward(img)

        # mask
        seg_mask = cv2.resize(preds_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}
        for label_idx in range(len(self.dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.dict, seg_mask.copy(), self.dict.class_seg_label_group[label_idx], self.dict.class_seg_label_group_2_id_map[self.dict.class_seg_label_group[label_idx]], self.dict.class_seg_label_group_threh_map[self.dict.class_seg_label_group[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info
    

class LPRSegPytorch(object):
    
    def __init__(self, pth_path, dict_name, input_shape=(128, 64)):

        self.model_path = pth_path
        self.input_shape = input_shape

        # self.dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + dict_name) 
        self.dict = importlib.import_module(dict_name) 

        self.model_init()


    def model_init(self):
        
        self.net = NovaLaneNet.LANENET_NOVT_TEST(num_classes=len(self.dict.class_seg_label)).cuda()
        state_dict=torch.load(self.model_path)['net']

        # load state
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()


    def model_forward(self, img):
        img_cuda = torch.from_numpy(img).unsqueeze(0).cuda()
        preds = self.net(img_cuda).detach().cpu().numpy()[0]
        preds_mask = np.argmax(preds, axis=0).astype(np.uint8)

        return preds_mask


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):
    
        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask = self.model_forward(img)

        # mask
        seg_mask = cv2.resize(preds_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}
        for label_idx in range(len(self.dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.dict, seg_mask.copy(), self.dict.class_seg_label_group[label_idx], self.dict.class_seg_label_group_2_id_map[self.dict.class_seg_label_group[label_idx]], self.dict.class_seg_label_group_threh_map[self.dict.class_seg_label_group[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return preds_mask, seg_mask, seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


class LPRSeg2HeadCaffe(object):
    
    def __init__(self, prototxt, model_path, city_dict_name, color_dict_name, input_shape=(128, 64), gpu_bool=False):
    
        self.prototxt = prototxt
        self.model_path = model_path
        self.input_shape = input_shape
        self.gpu_bool = gpu_bool

        self.city_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + city_dict_name) 
        self.color_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + color_dict_name) 

        self.seg_city_labels = self.city_dict.class_seg_label
        self.seg_color_labels = self.color_dict.class_seg_label

        self.ignore_boundary_ratio = 0.1

        self.model_init()


    def model_init(self):

        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)


    def model_forward(self, img):
        self.net.blobs['data'].data[...] = img

        preds_1 = self.net.forward()["prob"]
        preds_1 = np.squeeze(preds_1)
        preds_1 = preds_1 * (preds_1 > 0.5)
        preds_mask_1 = np.argmax(preds_1, axis=0).astype(np.uint8)

        preds_2 = self.net.forward()["prob_1"]
        preds_2 = np.squeeze(preds_2)
        preds_2 = preds_2 * (preds_2 > 0.9)
        preds_mask_2 = np.argmax(preds_2, axis=0).astype(np.uint8)
        return preds_mask_1, preds_mask_2


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask_1, preds_mask_2 = self.model_forward(img)

        # mask
        seg_mask_1 = cv2.resize(preds_mask_1, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        seg_mask_2 = cv2.resize(preds_mask_2, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # city
        for label_idx in range(len(self.city_dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.city_dict, seg_mask_1.copy(), \
                                            self.city_dict.class_seg_label_group[label_idx], \
                                            self.city_dict.class_seg_label_group_2_id_map[self.city_dict.class_seg_label_group[label_idx]], \
                                            self.city_dict.class_seg_label_group_threh_map[self.city_dict.class_seg_label_group[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        # color
        for label_idx in range(len(self.color_dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.color_dict, seg_mask_2.copy(), \
                                            self.color_dict.class_seg_label_group[label_idx], \
                                            self.color_dict.class_seg_label_group_2_id_map[self.color_dict.class_seg_label_group[label_idx]], \
                                            self.color_dict.class_seg_label_group_threh_map[self.color_dict.class_seg_label_group[label_idx]], \
                                            ignore_boundary_bool = True)
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre, ignore_boundary_bool=False):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 修理边界 boundary
        if ignore_boundary_bool:
            mask[0:int(mask.shape[0] * self.ignore_boundary_ratio), :] = 0
            mask[int(mask.shape[0] * (1 - self.ignore_boundary_ratio)):-1, :] = 0
            mask[:, 0:int(mask.shape[1] * self.ignore_boundary_ratio)] = 0
            mask[:, int(mask.shape[1] * (1 - self.ignore_boundary_ratio)):-1] = 0

        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


class LPRSeg2HeadPytorch(object):
    
    def __init__(self, pth_path, city_dict_name, color_dict_name, input_shape=(128, 64)):

        self.model_path = pth_path
        self.input_shape = input_shape

        self.city_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + city_dict_name) 
        self.color_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + color_dict_name) 

        self.seg_city_labels = self.city_dict.class_seg_label
        self.seg_color_labels = self.color_dict.class_seg_label

        # 颜色分割，边界易受车辆颜色影响
        self.ignore_boundary_ratio = 0.1

        self.model_init()


    def model_init(self):
        
        self.net = NovaLaneNet.LANENET_NOVT_2HEAD(num_classes_1_head=len(self.seg_city_labels), num_classes_2_head=len(self.seg_color_labels)).cuda()
        state_dict = torch.load(self.model_path)['net']

        # load state
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()


    def model_forward(self, img):
        img_cuda = torch.from_numpy(img).unsqueeze(0).cuda()
        preds_1, preds_2 = self.net(img_cuda)
        preds_1 = F.softmax(preds_1)
        preds_2 = F.softmax(preds_2)
        preds_1, preds_2 = preds_1.detach().cpu().numpy()[0], preds_2.detach().cpu().numpy()[0]

        preds_1 = preds_1 * (preds_1 > 0.5)
        preds_2 = preds_2 * (preds_2 > 0.9)
        preds_mask_1 = np.argmax(preds_1, axis=0).astype(np.uint8)
        preds_mask_2 = np.argmax(preds_2, axis=0).astype(np.uint8)

        return preds_mask_1, preds_mask_2


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):
    
        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask_1, preds_mask_2 = self.model_forward(img)

        # mask
        seg_mask_1 = cv2.resize(preds_mask_1, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        seg_mask_2 = cv2.resize(preds_mask_2, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # city
        for label_idx in range(len(self.city_dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.city_dict, seg_mask_1.copy(), \
                                            self.city_dict.class_seg_label_group[label_idx], \
                                            self.city_dict.class_seg_label_group_2_id_map[self.city_dict.class_seg_label_group[label_idx]], \
                                            self.city_dict.class_seg_label_group_threh_map[self.city_dict.class_seg_label_group[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        # color
        for label_idx in range(len(self.color_dict.class_seg_label_group)):
            bboxes, info = self.get_bboxes(self.color_dict, seg_mask_2.copy(), \
                                            self.color_dict.class_seg_label_group[label_idx], \
                                            self.color_dict.class_seg_label_group_2_id_map[self.color_dict.class_seg_label_group[label_idx]], \
                                            self.color_dict.class_seg_label_group_threh_map[self.color_dict.class_seg_label_group[label_idx]], \
                                            ignore_boundary_bool = True)
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre, ignore_boundary_bool=False):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 修理边界 boundary
        if ignore_boundary_bool:
            mask[0:int(mask.shape[0] * self.ignore_boundary_ratio), :] = 0
            mask[int(mask.shape[0] * (1 - self.ignore_boundary_ratio)):-1, :] = 0
            mask[:, 0:int(mask.shape[1] * self.ignore_boundary_ratio)] = 0
            mask[:, int(mask.shape[1] * (1 - self.ignore_boundary_ratio)):-1] = 0

        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


class LPRSegColorClassCaffe(object):
    
    def __init__(self, prototxt, model_path, city_dict_name, color_dict_name, input_shape=(128, 64), gpu_bool=False, city_bool=True, color_bool=True):
    
        self.prototxt = prototxt
        self.model_path = model_path
        self.input_shape = tuple(input_shape)
        self.gpu_bool = gpu_bool
        self.city_bool = city_bool
        self.color_bool = color_bool

        if self.city_bool:
            self.city_dict = importlib.import_module( city_dict_name ) 
            self.seg_city_labels = self.city_dict.class_seg_label
        
        if self.color_bool:
            self.color_dict = importlib.import_module( color_dict_name ) 
            self.seg_color_labels = self.color_dict.class_seg_label

        # 颜色分割，边界裁剪，便于画图
        self.ignore_boundary_ratio = 0.1

        self.model_init()


    def model_init(self):

        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)


    def model_forward(self, img):
        self.net.blobs['data'].data[...] = img

        preds_1 = self.net.forward()["prob"]
        preds_1 = np.squeeze(preds_1)
        preds_1 = preds_1 * (preds_1 > 0.5)
        preds_mask_1 = np.argmax(preds_1, axis=0).astype(np.uint8)

        preds_2 = self.net.forward()["prob_1"]
        preds_2 = np.squeeze(preds_2)
        preds_2 = preds_2 * (preds_2 > 0.75)
        preds_mask_2 = np.argmax(preds_2, axis=0).astype(np.uint8)
        return preds_mask_1, preds_mask_2


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask_1, preds_mask_2 = self.model_forward(img)

        # mask
        seg_mask_1 = cv2.resize(preds_mask_1, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        seg_mask_2 = cv2.resize(preds_mask_2, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # city
        if self.city_bool:
            for label_idx in range(len(self.city_dict.class_seg_label_group)):
                bboxes, info = self.get_bboxes(self.city_dict, seg_mask_1.copy(), \
                                                self.city_dict.class_seg_label_group[label_idx], \
                                                self.city_dict.class_seg_label_group_2_id_map[self.city_dict.class_seg_label_group[label_idx]], \
                                                self.city_dict.class_seg_label_group_threh_map[self.city_dict.class_seg_label_group[label_idx]])
                seg_bbox.update(bboxes)
                seg_info.update(info)       

        # color
        if self.color_bool:
            for label_idx in range(len(self.color_dict.class_seg_label_group)):
                bboxes, info = self.get_bboxes(self.color_dict, seg_mask_2.copy(), \
                                                self.color_dict.class_seg_label_group[label_idx], \
                                                self.color_dict.class_seg_label_group_2_id_map[self.color_dict.class_seg_label_group[label_idx]], \
                                                self.color_dict.class_seg_label_group_threh_map[self.color_dict.class_seg_label_group[label_idx]], \
                                                ignore_boundary_bool = True)
                seg_bbox.update(bboxes)
                seg_info.update(info)       

        return seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre, ignore_boundary_bool=False):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 修理边界 boundary
        if ignore_boundary_bool:
            mask[0:int(mask.shape[0] * self.ignore_boundary_ratio), :] = 0
            mask[int(mask.shape[0] * (1 - self.ignore_boundary_ratio)):-1, :] = 0
            mask[:, 0:int(mask.shape[1] * self.ignore_boundary_ratio)] = 0
            mask[:, int(mask.shape[1] * (1 - self.ignore_boundary_ratio)):-1] = 0

        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


class LPRSegColorClassPytorch(object):
    
    def __init__(self, pth_path, city_dict_name, color_dict_name, input_shape=(128, 64), city_bool=True, color_bool=True):

        self.model_path = pth_path
        self.input_shape = tuple(input_shape)

        self.city_bool = city_bool
        self.color_bool = color_bool

        if self.city_bool:
            self.city_dict = importlib.import_module( city_dict_name ) 
            self.seg_city_labels = self.city_dict.class_seg_label
        
        if self.color_bool:
            self.color_dict = importlib.import_module( color_dict_name ) 
            self.seg_color_labels = self.color_dict.class_seg_label

        self.city_dict = importlib.import_module( city_dict_name ) 
        self.color_dict = importlib.import_module( color_dict_name ) 

        # 颜色分割，边界裁剪，便于画图
        self.ignore_boundary_ratio = 0.1
        
        self.model_init()


    def model_init(self):
        
        self.net = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=len(self.seg_city_labels), num_classes_2_head=len(self.seg_color_labels)).cuda()
        state_dict = torch.load(self.model_path)['net']

        # load state
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()


    def model_forward(self, img):
        img_cuda = torch.from_numpy(img).unsqueeze(0).cuda()
        preds_1, preds_2 = self.net(img_cuda)
        preds_1 = F.softmax(preds_1)
        preds_2 = F.softmax(preds_2)
        preds_1, preds_2 = preds_1.detach().cpu().numpy()[0], preds_2.detach().cpu().numpy()[0]

        preds_1 = preds_1 * (preds_1 > 0.5)
        preds_2 = preds_2 * (preds_2 > 0.75)
        preds_mask_1 = np.argmax(preds_1, axis=0).astype(np.uint8)
        preds_mask_2 = np.argmax(preds_2, axis=0).astype(np.uint8)

        return preds_mask_1, preds_mask_2


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img


    def run(self, img):
    
        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        preds_mask_1, preds_mask_2 = self.model_forward(img)

        # mask
        seg_mask_1 = cv2.resize(preds_mask_1, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        seg_mask_2 = cv2.resize(preds_mask_2, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # bbox, info
        seg_bbox = {}
        seg_info = {}

        # city
        if self.city_bool:
            for label_idx in range(len(self.city_dict.class_seg_label_group)):
                bboxes, info = self.get_bboxes(self.city_dict, seg_mask_1.copy(), \
                                                self.city_dict.class_seg_label_group[label_idx], \
                                                self.city_dict.class_seg_label_group_2_id_map[self.city_dict.class_seg_label_group[label_idx]], \
                                                self.city_dict.class_seg_label_group_threh_map[self.city_dict.class_seg_label_group[label_idx]])
                seg_bbox.update(bboxes)
                seg_info.update(info)       

        # color
        if self.city_bool:
            for label_idx in range(len(self.color_dict.class_seg_label_group)):
                bboxes, info = self.get_bboxes(self.color_dict, seg_mask_2.copy(), \
                                                self.color_dict.class_seg_label_group[label_idx], \
                                                self.color_dict.class_seg_label_group_2_id_map[self.color_dict.class_seg_label_group[label_idx]], \
                                                self.color_dict.class_seg_label_group_threh_map[self.color_dict.class_seg_label_group[label_idx]], \
                                                ignore_boundary_bool = True)
                seg_bbox.update(bboxes)
                seg_info.update(info)       

        return preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox, seg_info


    def get_bboxes(self, dict_name, mask, label, mask_id_list, mask_seg_thre, ignore_boundary_bool=False):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label] = 'none'
        res_dict = {}

        mask_seg_thre = ( mask_seg_thre / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 修理边界 boundary
        if ignore_boundary_bool:
            mask[0:int(mask.shape[0] * self.ignore_boundary_ratio), :] = 0
            mask[int(mask.shape[0] * (1 - self.ignore_boundary_ratio)):-1, :] = 0
            mask[:, 0:int(mask.shape[1] * self.ignore_boundary_ratio)] = 0
            mask[:, int(mask.shape[1] * (1 - self.ignore_boundary_ratio)):-1] = 0

        # 对每个 标签 计数
        for mask_id in dict_name.id_2_mask_name_dict.keys():
            if mask_id in mask_id_list:
                res_dict[mask_id] = (mask == mask_id).sum()

        # 挑选最大 标签
        if max(res_dict.values()) > 0:
            res_id = max(res_dict, key=res_dict.get)

            # 标签 满足 大于 mask_seg_thre 限定条件
            if res_dict[res_id] > mask_seg_thre:
                
                # 取最大连通域
                mask[mask != res_id] = 0
                nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                bkg_id = 0
                for row in range(stats.shape[0]):
                    if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                        bkg_id = row
                stats_no_bg = np.delete(stats, bkg_id, axis=0)
                stats_no_bg_idx = stats_no_bg[:, 4].argmax()
                bboxes[dict_name.id_2_mask_name_dict[res_id]] = [stats_no_bg[stats_no_bg_idx]]
                info[label] = dict_name.id_2_mask_name_dict[res_id]
                
        return bboxes, info


if __name__ == '__main__': 

    # img = cv2.imread("/mnt/huanyuan/temp/test/0000000000000000-200215-132912-135912-00000D000210 03303-00_uae_sharjah_green_double_2#57262.jpg")
    img = cv2.imread("/mnt/huanyuan/temp/test/137_none_abudhabi_trp_none_Double_2_84772_0.jpg")

    # # torch
    # seg_pth_path = "/mnt/huanyuan/model/image/lanenet/seg_city_cartype_kind_num_zd_20230209/LaneNetNova.pth"
    # lpr_seg = LPRSegPytorch(seg_pth_path, "dataset_zd_dict")
    # preds_mask, seg_mask, seg_bbox_torch, seg_info_torch = lpr_seg.run(img)

    # # onnx
    # seg_onnx_path = "/mnt/huanyuan/model/image/lanenet/seg_city_cartype_kind_num_zd_20230209/zd_seg_0209.onnx"
    # lpr_seg = LPRSegOnnx(seg_onnx_path, "dataset_zd_dict")
    # seg_bbox_onnx, seg_info_onnx = lpr_seg.run(img)

    # # caffe
    # seg_caffe_prototxt = "/mnt/huanyuan/model/image/lanenet/seg_city_cartype_kind_num_zd_20230209/zd_seg_0209.prototxt"
    # seg_caffe_model = "/mnt/huanyuan/model/image/lanenet/seg_city_cartype_kind_num_zd_20230209/zd_seg_0209.caffemodel"
    # lpr_seg = LPRSegCaffe(seg_caffe_prototxt, seg_caffe_model, "dataset_zd_dict")
    # seg_bbox_caffe, seg_info_caffe = lpr_seg.run(img)

    # print(seg_bbox_torch)
    # print(seg_bbox_onnx)
    # print(seg_bbox_onnx)
    
    # torch
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/LaneNetNova2Head.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/LaneNetNova2Head_best.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/LaneNetNova2Head.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/LaneNetNova2Head_best.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/LaneNetNova2Head.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/LaneNetNova2Head_best.pth"
    # seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230714/LaneNetNova2Head.pth"
    seg_pth_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230714/LaneNetNova2Head_best.pth"
    lpr_seg = LPRSegColorClassPytorch(seg_pth_path, "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city", "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color")
    preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox_torch, seg_info_torch = lpr_seg.run(img)

    # caffe
    # seg_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/zd_seg_city_class_color_20230629.prototxt"
    # seg_caffe_model = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230629/zd_seg_city_class_color_20230629.caffemodel"
    # seg_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/zd_seg_city_class_color_20230703.prototxt"
    # seg_caffe_model = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/zd_seg_city_class_color_20230703.caffemodel"
    seg_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/zd_seg_city_class_color_20230705.prototxt"
    seg_caffe_model = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230705/zd_seg_city_class_color_20230705.caffemodel"
    seg_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230714/zd_seg_city_class_color_20230714.prototxt"
    seg_caffe_model = "/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_20230714/zd_seg_city_class_color_20230714.caffemodel"
    lpr_seg = LPRSegColorClassCaffe(seg_caffe_prototxt, seg_caffe_model, "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city", "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color")
    seg_bbox_caffe, seg_info_caffe = lpr_seg.run(img)

    # print(seg_bbox_torch)
    # print(seg_bbox_caffe)
    print(seg_info_torch)
    print(seg_info_caffe)
    