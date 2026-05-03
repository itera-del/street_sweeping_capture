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
ad_labels = ['#', '1', '2', '3', '4', '5', '6', '7', '8', '9','0','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I',
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
# filepath = r"\\192.168.80.4\CV_storage\F_public\students\chenshuo\guangdong_done"
write_path = "/mnt/ssd_disk_1/xnlin/xnlin_data/plate_zd_city2/"
file = glob.glob("/opt/8091/external/plate/license_plate/samplepics/UAE/check/fan/*.json")
len_file = len(file)
name="fan"
 
dict_color={1:"AD/", 2:"DUBAI/", 3:"SHJ/", 4:"RAK/",5:"AJMAN/",6:"UMMALQAIWAIN/",7:"FUJAIRAH/",8:"others/", 9:"others2/",10:"new/"}
dict_name={1:"Double/",2:"Single/"}

for i, file_name in enumerate(file):
    #print(i, "/", len_file, " file_name=", filepath+file_name)
    with io.open(file_name, "r",encoding="UTF-8") as f:
        try:
            data = json.load(f, encoding='utf-8')
            name_1=0
            name_2=0
            #img_name = data["imagePath"]
            #img_name=filepath+img_name
            img_name=file_name[:-4]+"jpg"
            num_img=num_img+1
            if (num_img%1000==0):
                print(num_img,len_file)
            img = cv2.imread(img_name, 1)
            try :
                img.shape
            except :
                print("no image")
                continue
            if (img.shape[1]*img.shape[0]<1500):
                continue
            #print("image is ok")
            for cell in data['shapes']:
                try:
                    T=cell['vidlan']
                except:
                    print("error")
                    continue
                old_name = cell["vehicleic"]
                if(len(old_name)<1):
                    print(f)
                    continue
                flag=0
                for i in old_name:
                    if(i not in ad_labels):
                        flag=1
                        print(i)
                if flag==1:
                    continue
                new_name=str(num_img)+name+"-"+old_name+".jpg"#+str(ocr_labels.index(old_name[7]))
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
                h = y2 - y1
                w = x2 - x1
                x1 = max(0, x1 + npr.randint(-0.08 * w, 1))
                x2 = min(img.shape[1], x2 + npr.randint(-1, 0.08 * w))
                y1 = max(0, y1 + npr.randint(-0.08 * h, 1))
                y2 = min(img.shape[0], y2 + npr.randint(-1, 0.08 * h))
                plate_img = img[y1:y2, x1:x2]
                if T=='Double_column':
                    name_1=1
                if T=='Single_column':
                    name_1=2
                if cell["region"]=='Abu_Dhabi' or cell["region"]=='AbuDhabi' or cell["region"]=='AD':
                    if cell['vidcolor']=='white_card':
                        name_2=10
                    else:
                        name_2=1
                elif cell["region"]=='Dubai':
                    name_2=2
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='Sharjah' or cell["region"]=='Shj':
                    name_2=3
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='Ras_Al_Khaimah' or cell["region"]=='RAK':
                    name_2=4
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='Ajman':
                    name_2=5
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='Umm_al_Qaiwain':
                    name_2=6
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='Fujairah':
                    name_2=7
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell['vidcolor']=='red_card' or cell['vidcolor']=='green_card' or cell['vidcolor']=='yellow_card':
                    name_2=9
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                elif cell["region"]=='unknownvehicleplate':
                    name_2=8
                    if(w/h>2.5):
                        name_1=2
                    else:
                        name_1=1
                else :
                    print(cell["region"])
                    name_2=8

                #cv2.imshow("plate_img", plate_img)
                #print(write_path+dict_name[name_1]+dict_color[name_2]+new_name)
                cv2.imwrite(write_path+dict_color[name_2]+dict_name[name_1]+new_name,plate_img)
                #print("OK")
            f.close()
        #     #print("-------------------------------------")
        except:
            print("Error : ", file_name)
            continue
