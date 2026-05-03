import cv2
from collections import OrderedDict
import math
import numpy as np
import os
import sys
import time
# import torch
# import torch.nn.functional as F

# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
# sys.path.insert(0, caffe_root + 'python')
import caffe

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
import Speech.API.Kws_weakup_Asr.impl.asr_decode_beamsearch as Decode_BeamSearch

# # sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
# sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
# import models.crnnv2 as crnnv2


ocr_labels_cn = ["-","皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']

ocr_labels_zd = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','#']

ocr_labels_brazil = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','#']

def greedy_decode( probs, blank_id = 0 ):
    
    prob_idxs = np.argmax( probs, axis=1 )
    
    first_pass = []
    first_pass_score = []
    for idx in range(len(prob_idxs)):
        prob_idx = prob_idxs[idx]
        if len(first_pass) == 0 or prob_idx != first_pass[-1]:
            first_pass.append( prob_idx )
            first_pass_score.append( probs[idx][prob_idx] )
    
    second_pass = []
    second_pass_score = []
    for idx in range(len(first_pass)):
        first_pass_idx = first_pass[idx]
        first_pass_score_idx = first_pass_score[idx]
        if first_pass_idx != blank_id:
            second_pass.append( first_pass_idx )
            second_pass_score.append( first_pass_score_idx )
    
    return second_pass, second_pass_score


class LPRCaffe(object):

    def __init__(self, prototxt_path, model_path, input_shape=(256, 64), ocr_labels=ocr_labels_cn, padding_bool=False, prefix_beam_search_bool=False, gpu_bool=False):
    
        self.prototxt_path = prototxt_path
        self.model_path = model_path
        self.input_shape = tuple(input_shape)
        self.ocr_labels = np.array(ocr_labels)
        self.prefix_beam_search_bool = prefix_beam_search_bool
        self.gpu_bool = gpu_bool
        self.padding_bool = padding_bool
        
        self.model_init()
    
    
    def model_init(self):

        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt_path, self.model_path, caffe.TEST)

        if self.prefix_beam_search_bool:

            self.asr_beamsearch = Decode_BeamSearch.BeamSearch(self.ocr_labels, beam_size=5)
            self.lm = None


    def model_forward(self, img):
        self.net.blobs['data'].data[...] = img
        preds = self.net.forward()["probs"]
        preds = np.transpose(np.squeeze(preds))
        return preds


    def preprocess(self, src):

        if self.padding_bool:
            # pad
            img_reshape = src[: src.shape[0]//4*4, : src.shape[1]//4*4]
            h, w = img_reshape.shape[0], img_reshape.shape[1]
            ratio = w * 1.0 / h
            max_wh_ratio = 256 * 1.0 / 64
            if ratio < max_wh_ratio:
                to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
                if (to_imgW - w < 32):
                    to_imgW = w + 32
                pad_img = np.ones((h, to_imgW), dtype=np.uint8)
                pad_img *= 255
                pad_img[:, 0:w] = img_reshape  
            else:
                pad_img = img_reshape
            src = pad_img

        output_img_path = "/home/huanyuan/share/crop.jpg";
        cv2.imwrite(output_img_path, src);

        img = cv2.resize(src, self.input_shape)
        img = img / 255.0
        return img


    def run(self, img):

        img = self.preprocess(img)
        img = img.astype(np.float32)

        preds = self.model_forward(img)

        if self.prefix_beam_search_bool:
            
            result_symbol_list = self.asr_beamsearch.prefix_beam_search(torch.from_numpy(preds).log(), lm=self.lm)
            result_ocr = ''.join(result_symbol_list)
            result_scors = 0.0
        else:

            # greedy 
            result_str, result_scors = greedy_decode(preds, blank_id=0)
            result_ocr = ''.join([self.ocr_labels[result_str[idx]] for idx in range(len(result_str))])

        return result_ocr, result_scors


class LPRPytorch(object):
    
    def __init__(self, pth_path, input_shape=(256, 64), ocr_labels=ocr_labels_cn, padding_bool=False, prefix_beam_search_bool=False):

        self.model_path = pth_path
        self.input_shape = input_shape
        self.ocr_labels = np.array(ocr_labels)
        self.prefix_beam_search_bool = prefix_beam_search_bool
        self.padding_bool = padding_bool

        self.model_init()
    
    
    def model_init(self):
        
        self.net = crnnv2.CRNNv3(nclass=len(self.ocr_labels)).cuda()
        # self.net = crnnv2.CRNNv6(nclass=len(self.ocr_labels)).cuda()
        state_dict=torch.load(self.model_path)

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

        if self.prefix_beam_search_bool:

            self.asr_beamsearch = Decode_BeamSearch.BeamSearch(self.ocr_labels, beam_size=5)
            self.lm = None


    def model_forward(self, img):   
        img_cuda = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
        preds = torch.squeeze(self.net(img_cuda))
        preds = F.softmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()
        
        return preds


    def preprocess(self, src):
        
        if self.padding_bool:
            # pad
            img_reshape = src[: src.shape[0]//4*4, : src.shape[1]//4*4]
            h, w = img_reshape.shape[0], img_reshape.shape[1]
            ratio = w * 1.0 / h
            max_wh_ratio = 256 * 1.0 / 64
            if ratio < max_wh_ratio:
                to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
                if (to_imgW - w < 32):
                    to_imgW = w + 32
                pad_img = np.ones((h, to_imgW), dtype=np.uint8)
                pad_img *= 255
                pad_img[:, 0:w] = img_reshape  
            else:
                pad_img = img_reshape
            src = pad_img

        img = cv2.resize(src, self.input_shape)
        img = img / 255.0
        return img


    def run(self, img):

        img = self.preprocess(img)
        img = img.astype(np.float32)

        preds = self.model_forward(img)

        if self.prefix_beam_search_bool:
            
            result_symbol_list = self.asr_beamsearch.prefix_beam_search(torch.from_numpy(preds).log(), lm=self.lm)
            result_ocr = ''.join(result_symbol_list)
            result_scors = 0.0
        else:

            # greedy 
            result_str, result_scors = greedy_decode(preds, blank_id=0)
            result_ocr = ''.join([self.ocr_labels[result_str[idx]] for idx in range(len(result_str))])

        return result_ocr, result_scors