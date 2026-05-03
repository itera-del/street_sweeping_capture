#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ Sept 30th 2019 xnlin    
# 
import cv2
import json
import os
import numpy as np
import glob
import io
import numpy.random as npr
ocr_labels = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂', '-']  # 33+24+10+1

print(ocr_labels)

num_img=0
def encode_plate(name, plate_str):
    new_name = []
    #global num_img
    num_img=num_img+1
    #plate_str = plate_str.strip(" ")
    print ("*****: ",plate_str)
    for i in plate_str:
        print(i)
        if i in ocr_labels:
            print ("yes")
            new_name.append(str(ocr_labels[i]))
    return str(name) + "-" + "_".join(new_name) + ".jpg"

write_path = "/yuanhuan/LicensePlateOCR/裁剪数据/ZG/"
filepath = "/yuanhuan/LicensePlateOCR/原始数据/ZG/"

os.chdir(filepath)
file = glob.glob("*.json")
len_file = len(file)
num2 = 0
for i, file_name in enumerate(file):
    with io.open(filepath + file_name, "r", encoding="UTF-8") as f:
            try:
                data = json.load(f, encoding='utf-8')
            except:
                print("error")
            img_name = filepath + file_name[:-4] + "jpg"
            img = cv2.imread(img_name, 1)
            num_img = num_img + 1
            if(num_img%1000==0):
                print(num_img , len_file)
            if img.shape == None:
                print("no image")
                continue
            for cell in data['shapes']:
                # 解析出车牌字符
                try:
                    T = cell['vidlan']
                    C = cell['vidcolor']
                    old_name = cell["vehicleic"]
                except:
                    continue
                # if '河' in old_name:
                #     continue
                # if '武' in old_name or '湖' in old_name or '山' in old_name:
                #     continue
                if '#' in old_name:
                    num2 = num2 + 1
                    print(num2)
                    old_name = old_name.replace('#', '')
                    old_name = old_name.replace('O', '0')
                    old_name = old_name.replace('泸', '沪')
                    old_name = old_name.replace('I', '1')
                    
                    if(len(old_name) == 7):
                            new_name=str(num_img)+"-"+str(ocr_labels.index(old_name[0]))+"_"+str(ocr_labels.index(old_name[1]))+"_"\
                                +str(ocr_labels.index(old_name[2]))+"_"+str(ocr_labels.index(old_name[3]))+"_"\
                                    +str(ocr_labels.index(old_name[4]))+"_"+str(ocr_labels.index(old_name[5]))+"_"\
                                        +str(ocr_labels.index(old_name[6]))+".jpg"#+str(ocr_labels.index(old_name[7]))
                            pts = np.array(cell["points"], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                            x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                            y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                            y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                            h=y2-y1
                            w=x2-x1
                            try:
                                x1 = max(0,x1+npr.randint(-0.08*w, 0.04*w))
                                x2 = min(img.shape[1],x2+npr.randint(-0.04*w, 0.08*w))
                                y1 = max(0,y1+npr.randint(-0.05*h, 0.05*h))
                                y2 = min(img.shape[0],y2+npr.randint(-0.05*h, 0.05*h))
                            except:
                                print("error")
                            plate_img = img[y1:y2, x1:x2]
                            if '挂' in old_name:
                                cv2.imwrite(write_path+'G/'+new_name,plate_img)
                            if num_img%9==0:
                                if C == 'blue_card':
                                    cv2.imwrite(write_path+'test/B/'+new_name,plate_img)
                                elif C == 'yellow_card':
                                    if T == 'Single_column':
                                        cv2.imwrite(write_path+'test/Y1/'+new_name,plate_img)
                                    else :
                                        cv2.imwrite(write_path+'test/Y2/'+new_name,plate_img)
                                else :
                                    cv2.imwrite(write_path+'test/O/'+new_name,plate_img)
                            else:
                                if C == 'blue_card':
                                    cv2.imwrite(write_path+'train/B/'+new_name,plate_img)
                                elif C == 'yellow_card':
                                    if T == 'Single_column':
                                        cv2.imwrite(write_path+'train/Y1/'+new_name,plate_img)
                                    else :
                                        cv2.imwrite(write_path+'train/Y2/'+new_name,plate_img)
                                else :
                                    cv2.imwrite(write_path+'train/O/'+new_name,plate_img)
                    if(len(old_name) == 8):
                            new_name=str(num_img)+"-"+str(ocr_labels.index(old_name[0]))+"_"+str(ocr_labels.index(old_name[1]))+"_"\
                                +str(ocr_labels.index(old_name[2]))+"_"+str(ocr_labels.index(old_name[3]))+"_"\
                                    +str(ocr_labels.index(old_name[4]))+"_"+str(ocr_labels.index(old_name[5]))+"_"\
                                        +str(ocr_labels.index(old_name[6]))+"_"+str(ocr_labels.index(old_name[7]))+".jpg"#
                            pts = np.array(cell["points"], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                            x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                            y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                            y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                            h=y2-y1
                            w=x2-x1
                            x1 = max(0,x1+npr.randint(-0.04*w, 0.04*w))
                            x2 = min(img.shape[1],x2+npr.randint(-0.04*w, 0.04*w))
                            y1 = max(0,y1+npr.randint(-0.05*h, 0.05*h))
                            y2 = min(img.shape[0],y2+npr.randint(-0.05*h, 0.05*h))
                            plate_img = img[y1:y2, x1:x2]
                            if num_img%9==0:
                                cv2.imwrite(write_path+'test/N/'+new_name,plate_img)
                            else:
                                cv2.imwrite(write_path+'train/N/'+new_name,plate_img)
