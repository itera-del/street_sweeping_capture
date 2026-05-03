import os
import cv2
import sys
import json
import numpy as np
from glob import glob 
from tqdm import tqdm
import random
caffe_root = '/home/workspace/xnlin/MobileNet-SSD-Focal-loss/'
sys.path.insert(0, caffe_root + 'python')    
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
net_file = '/home/workspace/xnlin/pytorch/visionlab/LPR/tools/FPN_RFB_5class_prior.prototxt'
caffe_model = '/home/workspace/xnlin/pytorch/visionlab/LPR/tools/best_0.747_SSD_VGG_FPN_RFB_VOC(1).caffemodel'
net = caffe.Net(net_file, caffe_model, caffe.TEST)
def preprocess(src):
    img = cv2.resize(src, (300, 300)).astype(np.float32)
    rgb_mean = np.array((104, 117, 123), dtype=np.int)
    img -= rgb_mean
    img = img.astype(np.float32)
    return img
def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
    clsi = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, clsi)

cls_color = {'inside':       [1, 1, 1],
             'outside':    [2, 2, 2],
             'ouinside':    [2, 2, 2],
            }

save_dir = '/home/workspace/xnlin/xnlin_data/BeltSeg/train/full'
data_dir = '/home/workspace/xnlin/xnlin_data/BeltSeg/full'
if not os.path.exists(save_dir+"/src/"):
    os.mkdir(save_dir+"/src/")
if not os.path.exists(save_dir+"/mask/"):
    os.mkdir(save_dir+"/mask/")
if not os.path.exists(save_dir+"/mask_img/"):
    os.mkdir(save_dir+"/mask_img/")
img_list = glob(data_dir + '/*/*.jpg')
jsn_list = glob(data_dir + '/*/*.json')

img_list.sort()
jsn_list.sort()

print(len(img_list))
print(len(jsn_list))
j=10000
N = len(jsn_list)
for i in tqdm(range(0, N)):
    jsn_path = jsn_list[i]
    img_path = jsn_path.replace('json', 'jpg')
    j=j+1
    src_save_path  = '{}/src/{}_src.jpg'.format(save_dir, str(j).zfill(5))
    mask_save_path = '{}/mask/{}_mask.png'.format(save_dir, str(j).zfill(5))
    mask_img_path  = '{}/mask_img/{}_maskimg.png'.format(save_dir, str(j).zfill(5))
    origimg = cv2.imread(img_path)
    try :
        mask = np.zeros(origimg.shape, dtype=origimg.dtype)
    except:
        print(img_path)
        continue
    with open(jsn_path, 'r') as fp:
        json_data = json.load(fp)
        for shape in json_data['shapes']:
            try:
                color = cls_color[shape['label']]
            except:
                continue
            contours = [np.array(shape['points'])]
            mask = cv2.drawContours(mask, contours, -1, color, cv2.FILLED)
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, clsi = postprocess(origimg, out)
    for i in range(len(box)):
        if conf[i] > 0.4 and clsi[i] == 4:
            left = box[i][0]
            top = box[i][1]
            right = box[i][2]
            bottom = box[i][3]
    w = right-left
    h = bottom - top
    size= max(250,int (random.randint(9,12) * max(w,h)/10))
    right = min(right+random.randint(-10, 10),origimg.shape[1])
    bottom = min(bottom+random.randint(-10, 10),origimg.shape[0])
    left = max(2,right-size)
    top = max(2,bottom-size)
    if (right<left or bottom<top or right>origimg.shape[1] or bottom>origimg.shape[0]):
        continue
    cv2.imwrite(mask_save_path, mask[top:bottom,left:right,:])
    cv2.imwrite(src_save_path, origimg[top:bottom,left:right,:])
            
    mask_img = cv2.addWeighted(origimg[top:bottom,left:right,:], 1, mask[top:bottom,left:right,:], 0.5, 0)
    cv2.imwrite(mask_img_path, mask_img)
    # img = origimg
    # with open(jsn_path, 'r') as fp:
    #     json_data = json.load(fp)
    #     for shape in json_data['shapes']:
    #         try:
    #             color = cls_color[shape['label']]
    #         except:
    #             print(shape['label'])
    #             continue
    #         contours = [np.array(shape['points'])]
    #         mask = cv2.drawContours(mask, contours, -1, color, cv2.FILLED)
    # cv2.imwrite(mask_save_path, mask)
    # cv2.imwrite(src_save_path, img)
            
    # mask_img = cv2.addWeighted(img, 1, mask, 0.5, 0)
    # cv2.imwrite(mask_img_path, mask_img)