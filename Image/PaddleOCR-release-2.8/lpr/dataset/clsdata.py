
import os
import cv2
import torch
import numpy as np
from dataset import aug
from random import shuffle
import torchvision.transforms as transforms
from utils import *
    
class HardSampleLoader(torch.utils.data.Dataset):
    def __init__(self, img_path, img_size=(128, 128), training=False):
        self.img_list = get_datalist(img_path)
        self.img_size = img_size
        self.training = training
        self.toTensor = transforms.ToTensor()
        shuffle(self.img_list)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_info = self.img_list[idx].strip()
        img_path, label = img_info.split(',')
        label = int(label)
        img = cv2.imread(img_path)
        # print(img.shape)
        # img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        if self.training:
            img = aug.imgaug(img)
        else :
            img = aug.imgaugtest(img)
        try:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        except:
            print ("*********************************{}".format(img_path))
            return self[idx+1]
        img = self.toTensor(img)
        return {"img": img, "label": label}

