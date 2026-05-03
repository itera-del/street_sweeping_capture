import os
import cv2
import math
import numpy as np
from aug import do_aug
from PIL import Image, ImageFont, ImageDraw

font1 = ImageFont.truetype("font/Dubai-Medium.otf", 110, 0)
font1_= ImageFont.truetype("font/Dubai-Light.otf", 105, 0)
font2 = ImageFont.truetype("font/Dubai-Medium.otf", 150, 0)

class plate_str(object):
    def __init__(self):
        pass
    def str1(self):
        ### up: 1~2, 1~19
        str1 = []
        n = np.random.randint(1, 20)
        str1.append(str(n))
        return str1
    def str2(self):
        ### down: 4~5
        str2 = []
        length = np.random.choice([4,5], p=(0.5,0.5))
        for i in range(length):
            n = np.random.randint(0, 10)
            str2.append(str(n))
        return str2

    def get_label(self):
        s1, s2 = self.str1(), self.str2()
        label = s1+['-']+s2
        label = ''.join(label)
        return [''.join(s1), ''.join(s2)], label

def gen_plates_double(N, path='imgs'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plate = plate_str()
    for i in range(N):
        s, p = plate.get_label()
        board = BOARD.copy()
        board = Image.fromarray(board)
        pixel0 = n = np.random.randint(0, 60)
        pixel1 = n = np.random.randint(30, 120)
        pixel255 = n = np.random.randint(200, 255)
        if len(s[0])==1:
            org1 = 290
        elif len(s[0])==2:
            org1 = 260
        ImageDraw.Draw(board).text((org1, -20), s[0], (pixel1,pixel1,pixel1), font=font1)
        ImageDraw.Draw(board).text((org1+2, -16), s[0], (pixel255,pixel255,pixel255), font=font1_)
        
        if len(s[1])==4:
            org2 = 160
        elif len(s[1])==5:
            org2 = 120
        ImageDraw.Draw(board).text((org2, 100), s[1], (pixel0,pixel0,pixel0), font=font2)
        
        board = np.array(board)
        cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = do_aug(board)
        
        # # view
        # cv2.imshow('', board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite('{}/{}night-{}.jpg'.format(path, str(i), p), board)

if __name__ == '__main__':
    
    BOARD = cv2.imread('src_board/plate_double.png')
    gen_plates_double(2000, 'imgs_double')
















