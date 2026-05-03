import os
import cv2
import math
import random
import numpy as np 
from aug import do_aug
from PIL import Image, ImageFont, ImageDraw

font1 = ImageFont.truetype("font/platech.ttf", 120, 0)
font1l = ImageFont.truetype("font/platech.ttf", 130, 0)
font2 = ImageFont.truetype("font/platechar.ttf", 180, 0)

font1s = ImageFont.truetype("font/platech.ttf", 100, 0)
font2s = ImageFont.truetype("font/platechar.ttf", 150, 0)
font2l = ImageFont.truetype("font/platechar.ttf", 190, 0)

TABLE = [
         '-' , # placeholder 0
         "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", 
         "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", 
         "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", # 33
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', # 24, except I,O
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 10
         '挂', 
         ]

ocr_labels = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂', '-']  # 33+24+10+1
class plate_str(object):
    def __init__(self):
        pass
    def provence(self):
        str1 = []
        n = np.random.randint(1, 32)
        str1.append(str(TABLE[n]))
        return str1
    
    def city(self):
        str2 = []
        n = np.random.randint(34, 58)
        str2.append(str(TABLE[n]))
        return str2
    
    def alp_num(self):
        str3 = []
        n_alpha = np.random.choice([0,1,2])
        count_alpha = 0
        for i in range(5):
            if count_alpha < n_alpha:
                n = np.random.randint(34, 58)
                count_alpha += 1
            else:
                n = np.random.randint(58, 68)
            str3.append(str(TABLE[n]))
        random.shuffle(str3)
        
        if np.random.rand() >= 0.6:
            str3[-1] = str(TABLE[-1])
        return str3

    def get_label(self):
        p, c, n = self.provence(), self.city(), self.alp_num()
        label = p + c + n
        label = ''.join(label)
        return [''.join(p), ''.join(c), ''.join(n)], label

def dilate(img, ksize=1):
    img = np.array(img)
    tmp1 = img[140:-20, 10:-10, :]
    tmp2 = img[10:130, 300:400, :]

    k = np.ones((ksize, ksize), np.uint8)
    tmp1 = cv2.erode(tmp1, k)
    tmp2 = cv2.erode(tmp2, k+1)
    
    img[140:-20, 10:-10, :] = tmp1
    img[10:130, 300:400, :] = tmp2
    
    img = Image.fromarray(img)
    return img

def blur(img, k=3):
    img = np.array(img)
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = Image.fromarray(img)
    return img

def gen_plates_double(N, path='imgs'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plate = plate_str()
    for t in range(N):
        s, p = plate.get_label()
        board = BOARD.copy()
        board = Image.fromarray(board)
        ImageDraw.Draw(board).text((100, 5), s[0], (0,0,0), font=font1s)
        ImageDraw.Draw(board).text((320, -10), s[1], (0,0,0), font=font2s)
        
        if s[2][-1] == '挂':
            for i in range(4):
                ImageDraw.Draw(board).text((15+i*90, 120), s[2][i], (0,0,0), font=font2l)
            ImageDraw.Draw(board).text((375, 130), s[2][-1], (0,0,0), font=font1l)
        else:
            for i in range(5):
                ImageDraw.Draw(board).text((15+i*100, 120), s[2][i], (0,0,0), font=font2l)
        
        board = dilate(board, 4)
        board = blur(board, 3)
        
        board = np.array(board)
        cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = do_aug(board)
        old_name = p
        new_name=str(t)+"-"+str(ocr_labels.index(old_name[0]))+"_"+str(ocr_labels.index(old_name[1]))+"_"\
            +str(ocr_labels.index(old_name[2]))+"_"+str(ocr_labels.index(old_name[3]))+"_"\
                +str(ocr_labels.index(old_name[4]))+"_"+str(ocr_labels.index(old_name[5]))+"_"\
                    +str(ocr_labels.index(old_name[6]))+".jpg"
        img = Image.fromarray(board[..., ::-1])
        img.save('{}/{}'.format('/home/workspace/xnlin/xnlin_data/plate_cn/gen/Y2/', new_name))
        

if __name__ == '__main__':
    
    BOARD = cv2.imread('src_board/plate_double.png')
    gen_plates_double(5000, 'imgs_double')
















