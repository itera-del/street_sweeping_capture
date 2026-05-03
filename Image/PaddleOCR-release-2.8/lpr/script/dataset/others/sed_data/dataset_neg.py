import os
import cv2
import json
import random
import numpy as np
import numpy.random as npr
from glob import glob
from tqdm import tqdm
img_list = glob('/mnt/ssd_disk_1/xnlin/xnlin_data/cls/belt//neg/*jpg')
save_path = "/mnt/ssd_disk_1/xnlin/xnlin_data/BeltSeg/new/belt_off/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
random.shuffle(img_list)
print("{}".format(len(img_list)))
num = min(10000,len(img_list))
infos = []
name="belt_off"
now = 0
for i in tqdm(range(len(img_list))):
    try:
        if "changsha" not in img_list[i]:
            continue        
        img = cv2.imread(img_list[i])
        mask = np.zeros(img.shape, dtype=img.dtype)
        imgname = name + str(now)
        cv2.imwrite(save_path + imgname+".png", mask)
        cv2.imwrite(save_path + imgname+".jpg", img)
        imginfo = '{} {}\n'.format(save_path + imgname+".jpg", save_path + imgname+".png")
        infos.append(imginfo)
        now = now + 1
    except:
        print("{} error".format(img_list[i]))
        continue
with open('list.txt', 'w+') as f:
    for i in infos:
        f.write(i)
