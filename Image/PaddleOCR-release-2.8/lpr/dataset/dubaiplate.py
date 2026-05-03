#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ Mar 04th 2020 xnlin    
#
import math
import os
import cv2
import sys
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, sampler
import torchvision.transforms as transforms

from dataset import aug


def get_image_from_path(base_path):
    images = [os.path.join(base_path, i) for i in os.listdir(base_path) if (i.endswith('g')or(i.endswith('G')))]
    random.shuffle(images)
    return images

def get_image_from_paths(inputPathList):
    imgPathList = []
    for i in range(len(inputPathList)):
        print(inputPathList[i])
        images_list = get_image_from_path(inputPathList[i])
        imgPathList.extend(images_list)
    random.shuffle(imgPathList)
    return imgPathList

class DatasetPlate(Dataset):
    def __init__(self, imglistPath, inputSize , istrain):
        print("img******************", imglistPath)
        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList = get_image_from_paths(imglistPath)
        else:
            self.imgPathList = get_image_from_path(imglistPath)
        print("plate nums: ", len(self.imgPathList))
        self.num=0
        self.istrain = istrain
        self.inputSize = inputSize
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        label_list=[]
        self.num += 1
        # if(self.channel == 1):
        img = cv2.imread(self.imgPathList[index],0)
        # else:
        #     img = cv2.imread(self.imgPathList[index])
        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and self.imgPathList[index]\n", self.imgPathList[index])
            sys.exit()
        probability = np.random.rand(1,1)
        img=aug.imgaug(img)
        img=cv2.resize(img,(300,64))#(94,24)
        name=self.imgPathList[index].strip().split("-")
        labels_name=name[-1].strip().split(".")
        ##########################
        labels=labels_name[0].strip().split("#")
        labels2=''
        for i in range(len(labels)):
            if(labels[i]=="TAXI" or labels[i]=="DUBAI" or labels[i]=="POLICE" or labels[i]=="POICE" or labels[i]=="TXAI"or labels[i]=="TAXU"):
                continue
            labels2=labels2+''.join(labels[i])
        if (len(labels2)<1):
            print (labels2)
            print(self.imgPathList[index])
            return self[index + 1]
        ##########################
        img=self.toTensor(img)
        return img, labels2

class DatasetPlate2(Dataset):
    def __init__(self, imglistPath, inputSize , istrain):
        print("img******************", imglistPath)
        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList = get_image_from_paths(imglistPath)
        else:
            self.imgPathList = get_image_from_path(imglistPath)
        print("plate nums: ", len(self.imgPathList))
        self.num=0
        self.istrain = istrain
        self.inputSize = inputSize
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        label_list=[]
        self.num += 1
        # if(self.channel == 1):
        img = cv2.imread(self.imgPathList[index],0)
        # else:
        #     img = cv2.imread(self.imgPathList[index])
        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and self.imgPathList[index]\n", self.imgPathList[index])
            sys.exit()
        if(self.istrain==1):
            img=aug.imgaug(img)
        img=cv2.resize(img,(256,32))#(300,64)
        name=self.imgPathList[index].strip().split("-")
        labels_name=name[-1].strip().split(".")
        ##########################
        labels=labels_name[0]
        labels=labels.replace('TAXI#', '')
        labels=labels.replace('DUBAI#', '')
        labels=labels.replace('POLICE#', '')
        labels=labels.replace('POICE#', '')
        labels=labels.replace('TXAI#', '')
        labels=labels.replace('TAXU#', '')
        labels=labels.replace('TAXI#', '')
        labels=labels.replace('#DUBAI', '')
        labels=labels.replace('#POLICE', '')
        labels=labels.replace('#POICE', '')
        labels=labels.replace('#TXAI', '')
        labels=labels.replace('#TAXU', '')
        labels=labels.replace('TAXI', '')
        labels=labels.replace('DUBAI', '')
        labels=labels.replace('POLICE', '')
        labels=labels.replace('POICE', '')
        labels=labels.replace('TXAI', '')
        labels=labels.replace('TAXU', '')
        if (len(labels)<1):
            print (labels)
            print(self.imgPathList[index])
            return self[index + 1]
        ##########################
        img=self.toTensor(img)
        return img, labels

class DatasetPlate3(Dataset):
    def __init__(self, imglistPath, inputSize , istrain):
        print("img******************", imglistPath)
        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList = get_image_from_paths(imglistPath)
        else:
            self.imgPathList = get_image_from_path(imglistPath)
        print("plate nums: ", len(self.imgPathList))
        self.num=0
        self.istrain = istrain
        self.inputSize = inputSize
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        label_list=[]
        self.num += 1
        # if(self.channel == 1):
        img = cv2.imread(self.imgPathList[index],0)
        # else:
        #     img = cv2.imread(self.imgPathList[index])
        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and self.imgPathList[index]\n", self.imgPathList[index])
            sys.exit()
        probability = np.random.rand(1,1)
        if(self.istrain):
            img=aug.imgaug(img)
        img=cv2.resize(img,(256,32))#(256,32)
        name=self.imgPathList[index].strip().split("/")
        name2=name[-1].strip().split("__")
        name3 = name2[-1].strip().split("DUBAI-")
        labels_name=name3[-1].strip().split(".")
        labels=labels_name[0].strip().split("-")
        labels2=''
        for i in range(len(labels)):
            labels2=labels2+''.join(labels[i])
        img=self.toTensor(img)
        labels2=labels2.replace('#', '')
        labels2=labels2.replace(',', '')
        if len(labels2)<3:
            print(self.imgPathList[index])
        #print(labels2)
        return img, labels2

class DatasetPlateSet(Dataset):
    def __init__(self, imglistPath, image_set, gray_bool=True, padding_bool=True):
        super(DatasetPlateSet, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        
        self.data_dir_path = imglistPath
        self.image_set = image_set
        self.toTensor = transforms.ToTensor()
        self.gray_bool = gray_bool
        self.padding_bool = padding_bool

        self.createIndex()

        if image_set == 'train':
            self._shuffle()

        print("image_set: {}, plate nums: {}".format(image_set, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def createIndex(self):
        listfile = self.data_dir_path + "/ImageSets/Main/" + "{}.txt".format(self.image_set)
        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(line)
        
    def _shuffle(self):
        random.shuffle(self.img_list)

    def __getitem__(self, index):
        
        # load
        try:
            image = Image.open(self.img_list[index])
        except IOError:
            print(self.img_list[index])
            return self[index + 1]
        try:
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            if self.gray_bool:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #img= np.array(img, dtype=np.float32)
            else:
                pass
        except :
            print("img error: " + self.img_list[index])
            return self[index + 1]
        
        # check
        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none: " + self.img_list[index])
            return self[index + 1]

        if(self.image_set == 'train'):
            img = aug.imgaug(img)

        if self.padding_bool:
            # pad
            img_reshape = img[: img.shape[0]//4*4, : img.shape[1]//4*4]
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
            img = pad_img

        # resize
        img = cv2.resize(img, (256,64))

        labels = os.path.basename(self.img_list[index]).replace('.jpg', '').split('_')[-1]
        # labels = os.path.basename(self.img_list[index]).replace('.jpg', '').split('_')[-1][:3]
        # labels = os.path.basename(self.img_list[index]).replace('.jpg', '').split('_')[-1][3:]
        img=self.toTensor(img)

        # print("img_path: {}, label: {}".format(self.img_list[index], labels))
        return img, labels


class DatasetChinaPlate(Dataset):
    def __init__(self, imglistPath, inputSize, istrain):
        print("img******************", imglistPath)
        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList = get_image_from_paths(imglistPath)
        else:
            self.imgPathList = get_image_from_path(imglistPath)
        print("plate nums: ", len(self.imgPathList))
        self.channel = 3
        self.toTensor = transforms.ToTensor()
        self.istrain=istrain
        self.num=0
        self.ocr_labels = ["-","皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']


    def get_roi_img(self, img, roiRect):
        imgRoi = img[int(roiRect[1]) : int(roiRect[1]+roiRect[3]), int(roiRect[0]):int((roiRect[0]+roiRect[2]))].copy()
        return imgRoi 

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        label_list=[]
        self.num += 1
        # if(self.channel == 1):
        #     img = cv2.imread(self.imgPathList[index], 0)
        # else:
        #     img = cv2.imread(self.imgPathList[index],0)
        try:
            image = Image.open(self.imgPathList[index])
        except IOError:
            print(self.imgPathList[index])
            return self[index + 1]
        try:
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img= np.array(img, dtype=np.float32)
        except :
            print("img error! "+self.imgPathList[index])
            return self[index + 1]
        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! "+self.imgPathList[index])
            return self[index + 1]
            #sys.exit()
        if(self.istrain==1):
            img=aug.imgaug(img)
        img=cv2.resize(img,(256,64))
        name=self.imgPathList[index].strip().split("-")
        labels_name=name[-1].strip().split(".")
        ##########################
        labels=labels_name[0].strip().split("_")
        
        try :
            labels=''.join(self.ocr_labels[int(labels[i])+1] for i in range(len(labels)))
        except: 
            print(self.imgPathList[index])
            return self[index + 1]
        ##########################
        img=self.toTensor(img)
        #img = torch.from_numpy(img).type(torch.FloatTensor)
        return img, labels#labels_name[0]


class Str2Label(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet 
        self.dict  = {char:i for i, char in enumerate(alphabet)} 


    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        for item in text:
            #result = [self.dict[item[x]] for x in range(len(item))]
            i=0
            for char in item:
                char = char.upper()
                # if char=="A" or char=="B" or char=="C" or char=="D" or char=="E" or char=="F" or char=="G" or char=="H" or char=="I" or char=="J" or char=="K" or char=="L" or char=="M" or char=="N" or char=="O" or char=="P" or char=="Q" or char=="R" or char=="S" or char=="T" or char=="U" or char=="V" or char=="W" or char=="X" or char=="Y" or char=="Z":
                #     continue
                # if char=="a" or char=="b" or char=="c" or char=="d" or char=="e" or char=="f" or char=="g" or char=="h" or char=="i" or char=="j" or char=="k" or char=="l" or char=="m" or char=="n" or char=="o" or char=="p" or char=="q" or char=="r" or char=="s" or char=="t" or char=="u" or char=="v" or char=="w" or char=="x" or char=="y" or char=="z":
                #     continue
                try:
                    index = self.dict[char]
                except:
                    print (char, text)
                result.append(index)
                i=i+1
            length.append(i)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class Str2Labelchina(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet 
        self.dict  = {char:i for i, char in enumerate(alphabet)} 


    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        for item in text:
            #result = [self.dict[item[x]] for x in range(len(item))]
            i=0
            for char in item:
                if char=="#":
                    continue
                index = self.dict[char]
                result.append(index)
                i=i+1
            length.append(i)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts