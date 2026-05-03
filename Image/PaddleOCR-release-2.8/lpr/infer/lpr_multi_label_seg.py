import cv2
from collections import OrderedDict
import importlib
import numpy as np
import sys
import torch

sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
# sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
import models.NovaLaneNet as NovaLaneNet


class LPRSegColorPytorch(object):
    
    def __init__(self, pth_path, dict_name, input_shape=(128, 64)):

        self.model_path = pth_path
        self.input_shape = input_shape
        self.pred_thread = 0.5
        
        self.dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + dict_name) 

        self.model_init()


    def model_init(self):
        
        self.net = NovaLaneNet.LANENET_NOVT_TEST(num_classes=len(self.dict.class_seg_sigmoid_label)).cuda()
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
        pred = self.net(img_cuda)[0]
        pred_prob = pred.sigmoid()
        pred_prob = pred_prob.permute(1, 2, 0).contiguous()
        pred_prob = pred_prob.detach().cpu().numpy()

        return pred_prob


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
        pred_prob = self.model_forward(img)
        pred_mask = np.zeros(pred_prob.shape, dtype="uint8")
        pred_mask[pred_prob > self.pred_thread] = 1.0

        # mask
        seg_prob = cv2.resize(pred_prob, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        seg_mask = np.zeros(seg_prob.shape, dtype=seg_prob.dtype)
        seg_mask[seg_prob > self.pred_thread] = 1.0

        # bbox, info
        seg_bbox = {}
        seg_info = {}
        for label_idx in range(len(self.dict.class_seg_sigmoid_label)):
            bboxes, info = self.get_bboxes(seg_mask.copy(), label_idx, self.dict.class_seg_sigmoid_label[label_idx], self.dict.class_seg_label_sigmoid_threh_map[self.dict.class_seg_sigmoid_label[label_idx]])
            seg_bbox.update(bboxes)
            seg_info.update(info)       

        return pred_mask, seg_mask, seg_bbox, seg_info


    def get_bboxes(self, seg_mask, label_idx, label_name, label_num_threh):
        # return:
        #   bboxes：{ 'label': [[x, y, w, h]] }

        # init 
        bboxes = {}
        info = {}
        info[label_name] = 'none'
 
        label_num_threh = ( label_num_threh / (self.input_shape[0] * self.input_shape[1]) ) * (self.image_width * self.image_height)
        
        # 对 标签 计数
        res_value = (seg_mask[:,:,label_idx] == 1).sum()

        # 标签 满足 大于 label_num_threh 限定条件
        if res_value > label_num_threh:
            
            # 取最大连通域
            nums, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_mask[:,:,label_idx].astype("uint8"), connectivity=8)

            bkg_id = 0
            for row in range(stats.shape[0]):
                if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                    bkg_id = row
            stats_no_bg = np.delete(stats, bkg_id, axis=0)
            stats_no_bg_idx = stats_no_bg[:, 4].argmax()
            bboxes[label_name] = [stats_no_bg[stats_no_bg_idx]]
            info[label_name] = label_name
                
        return bboxes, info