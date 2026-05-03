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
ocr_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', '-']  # 33+24+10+1
ad_labels = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9','0','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I',
            'J', 'K', 'L', 'M', 'N', 'P', 'O','Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



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
    #assert len(new_name) == 7
    return str(name) + "-" + "_".join(new_name) + ".jpg"

print("********************************")
write_path = "/home/workspace/xnlin/xnlin_data/Mexican/train/"
file = glob.glob("/opt/8091/external/plate/license_plate/samplepics/Mexican/20210730/day/*.json")
len_file = len(file)
name="fan"
print(len(file))
dict_color={1:"/", 2:"/", 3:"/", 4:"/",5:"/"}
dict_name={1:"/",2:"/"}

for i, file_name in enumerate(file):
    #print(i, "/", len_file, " file_name=", filepath+file_name)
    with io.open(file_name, "r",encoding="UTF-8") as f:
            try :
                data = json.load(f, encoding='utf-8')
            except:
                continue
            name_1=0
            name_2=0
            #img_name=filepath+img_name
            img_name=file_name[:-4]+"jpg"
            num_img=num_img+1
            if (num_img%1000==0):
                print(num_img,len_file)
            img = cv2.imread(img_name, 1)
            for cell in data['shapes']:
                try :
                    img.shape
                    old_name = cell["vehicleic"]
                except:
                    print(file_name)
                    continue
                flag=0
                old_name=old_name.replace('-', '')
                for i in old_name:
                    if(i not in ad_labels):
                        flag=1
                        print(i)
                if flag==1:
                    continue
                new_name=str(num_img)+name+"__"+old_name+".jpg"#+str(ocr_labels.index(old_name[7]))
                #print("new_name=", new_name)

                # 解析得到四个角点坐标
                pts = np.array(cell["points"], np.int32)
                #print(pts)

                # 得到最小外接矩形的左上右下坐标点
                pts = pts.reshape((-1, 1, 2))
                x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
                y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
                h=y2-y1
                w=x2-x1
                x1 = max(0,x1+npr.randint(-0.08*w, 1))
                x2 = min(img.shape[1],x2+npr.randint(-1, 0.08*w))
                y1 = max(0,y1+npr.randint(-0.08*h, 1))
                y2 = min(img.shape[0],y2+npr.randint(-1, 0.08*h))
                plate_img = img[y1:y2, x1:x2]

                #cv2.imshow("plate_img", plate_img)
                #print(write_path+dict_name[name_1]+dict_color[name_2]+new_name)
                cv2.imwrite(write_path+new_name,plate_img)
                #print("OK")
            f.close()
        #     #print("-------------------------------------")

