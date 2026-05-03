import os
import cv2
import math
import numpy as np
from aug import do_aug
from PIL import Image, ImageFont, ImageDraw

m = np.random.randint(70, 80)
font1 = ImageFont.truetype("font/Dubai-Medium.otf", m, 0)
# font1_= ImageFont.truetype("font/Dubai-Light.otf", 105, 0)
font2 = ImageFont.truetype("font/Dubai-Medium.otf", 145, 0)

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
        length = np.random.choice([4,5], p=(0.3,0.7))
        for i in range(length):
            n = np.random.randint(0, 10)
            str2.append(str(n))
        return str2

    def get_label(self):
        s1, s2 = self.str1(), self.str2()
        label = s1+['#']+s2
        label = ''.join(label)
        return [''.join(s1), ''.join(s2)], label

def gen_plates_single(N, path='imgs'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plate = plate_str()
    for i in range(N):
        s, p = plate.get_label()
        board = BOARD.copy()
        board = Image.fromarray(board)
        
        if len(s[0])==1:
            org1 = 40
        elif len(s[0])==2:
            org1 = 20
        if len(s[1])==4:
            org2 = 425
        elif len(s[1])==5:
            org2 = 375
            
        ImageDraw.Draw(board).text((org1, -15), s[0], (0,0,0), font=font1)
        # ImageDraw.Draw(board).text((org1+2, -6), s[0], (255,255,255), font=font1_)
        ImageDraw.Draw(board).text((org2, -30), s[1], (0,0,0), font=font2)
        
        board = np.array(board)
        cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = do_aug(board)
        
        # # view
        # cv2.imshow('', board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('{}/{}night-{}.jpg'.format(path, str(i),p), board)




if __name__ == '__main__':

    BOARD = cv2.imread('src_board/plate_single.jpg')
    gen_plates_single(3000, 'imgs_single')
















