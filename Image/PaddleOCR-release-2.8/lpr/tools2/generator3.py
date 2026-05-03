import os
import cv2
import math
import numpy as np

### augs
def random_gaussian(img):
    k = np.random.choice([1,3,5,7,9,11,13,15])
    img = cv2.GaussianBlur(img,(k,k),0)
    return img

def motion_blur(image):
    degree = np.random.randint(2, 7)
    angle  = np.random.randint(0, 180)
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def addnoise(img):
    noise = np.random.uniform(-16, 16, size=img.shape)
    noi_img = np.clip(img+noise, 0, 255)
    noi_img = np.array(noi_img, dtype=np.uint8)
    return noi_img

def brightness(img):
    alpha = np.random.uniform(0.5, 1.0)
    beta  = np.random.uniform(0.0, 60.0)
    blank = np.zeros(img.shape, img.dtype)
    img = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return img

def arealight(img):
    h, w, _ = img.shape
    light = cv2.imread('light.png')
    light = cv2.resize(light, (max(h,w)*2, max(h,w)*2))
    hoffset = np.random.randint(0, w-h)
    woffset = np.random.randint(0, w)
    light = light[hoffset:hoffset+h, woffset:woffset+w, ...]
    w = np.random.uniform(0.3, 0.6)
    img = cv2.addWeighted(img, w, light, 1-w, 0)
    return img

def mosaic(img):
    rows, cols, _ = img.shape
    p = [int(rows*0.6), int(cols*0.5)]
    xmin = np.random.randint(0, cols-p[1])
    ymin = np.random.randint(0, rows-p[0])
    xmax = xmin + p[1]
    ymax = ymin + p[0]
    block = img[ymin:ymax, xmin:xmax, :]
    rate = np.random.randint(2, 8)
    block = cv2.resize(block, (p[1]//rate, p[0]//rate), interpolation=cv2.INTER_NEAREST)
    block = cv2.resize(block, (p[1], p[0]), interpolation=cv2.INTER_NEAREST)
    img[ymin:ymax, xmin:xmax, :] = block
    return img

def squeeze(img):
    h, w, _ = img.shape
    rate = np.random.uniform(0.3, 1)
    
    img = cv2.resize(img, (int(w*rate), int(h*rate)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img

def PerspectiveTransform(img):
    rows, cols, _ = img.shape
    h_range = rows*0.15
    w_range = cols*0.05
    p1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    p2 = np.float32([[np.random.uniform(0, w_range)         , np.random.uniform(0, h_range)], 
                     [np.random.uniform(cols-w_range, cols) , np.random.uniform(0, h_range)], 
                     [np.random.uniform(0, w_range)         , np.random.uniform(rows-h_range, rows)], 
                     [np.random.uniform(cols-w_range, cols) , np.random.uniform(rows-h_range, rows)]
                     ])
    M = cv2.getPerspectiveTransform(p1,p2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def addmud(img):
    rows, cols, _ = img.shape
    mud = cv2.imread('mud.jpg')
    h, w, _ = mud.shape
    rate = np.random.uniform(0.2, 0.6)
    hrange = int(h*rate)
    wrange = int(w*rate)
    xmin = np.random.randint(0, w-wrange)
    ymin = np.random.randint(0, h-hrange)
    block = mud[ymin:ymin+hrange, xmin:xmin+wrange, :]
    block = cv2.resize(block, (cols, rows))
    w = np.random.uniform(0.4, 0.6)
    img = cv2.addWeighted(img, w, block, (1-w), 0)
    
    return img

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
        label = s1+['#']+s2
        label = ''.join(label)
        return [''.join(s1), ''.join(s2)], label

def gen_plates(N, path='imgs'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plate = plate_str()
    for i in range(N):
        s, p = plate.get_label()
        board = BOARD.copy()
        pixel0 = n = np.random.randint(0, 40)
        pixel1 = n = np.random.randint(40, 100)
        pixel255 = n = np.random.randint(200, 255)
        if len(s[0])==1:
            cv2.putText(board, text=s[0], org=(40,115), fontFace=FONT, fontScale=L_scale, color=(pixel1,pixel1,pixel1), thickness=L_thickness)
            cv2.putText(board, text=s[0], org=(40,115), fontFace=FONT, fontScale=L_scale, color=(pixel255,pixel255,pixel255), thickness=tk_inside)
        elif len(s[0])==2:
            cv2.putText(board, text=s[0], org=(10,115), fontFace=FONT, fontScale=L_scale, color=(pixel1,pixel1,pixel1), thickness=L_thickness)
            cv2.putText(board, text=s[0], org=(10,115), fontFace=FONT, fontScale=L_scale, color=(pixel255,pixel255,pixel255), thickness=tk_inside)

        if len(s[1])==4:
            cv2.putText(board, text=s[1], org=(320,120), fontFace=FONT, fontScale=R_scale, color=(pixel0,pixel0,pixel0), thickness=R_thickness)
        elif len(s[1])==5:
            cv2.putText(board, text=s[1], org=(280,120), fontFace=FONT, fontScale=R_scale, color=(pixel0,pixel0,pixel0), thickness=R_thickness)
        
        cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = brightness(board)
        board = squeeze(board)
        # board = addmud(board)
        board = random_gaussian(board)
        board = motion_blur(board)
        board = addnoise(board)
        
        #board = arealight(board)
        board = mosaic(board)
        board = mosaic(board)
        board = PerspectiveTransform(board)
        
        # # view
        # cv2.imshow('', board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite('{}/{}night-{}.jpg'.format(path,str(i), p), board)


def gen_plates(N, path='imgs'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plate = plate_str()
    for i in range(N):
        s, p = plate.get_label()
        board = BOARD.copy()
        pixel0 = n = np.random.randint(0, 60)
        pixel1 = n = np.random.randint(30, 120)
        pixel255 = n = np.random.randint(200, 255)
        if len(s[0])==1:
            cv2.putText(board, text=s[0], org=(290,100), fontFace=FONT, fontScale=UP_scale, color=(pixel1,pixel1,pixel1), thickness=UP_thickness)
            cv2.putText(board, text=s[0], org=(290,100), fontFace=FONT, fontScale=UP_scale, color=(pixel255,pixel255,pixel255), thickness=tk_inside)
        elif len(s[0])==2:
            cv2.putText(board, text=s[0], org=(270,100), fontFace=FONT, fontScale=UP_scale, color=(pixel1,pixel1,pixel1), thickness=UP_thickness)
            cv2.putText(board, text=s[0], org=(270,100), fontFace=FONT, fontScale=UP_scale, color=(pixel255,pixel255,pixel255), thickness=tk_inside)
            
        if len(s[1])==4:
            cv2.putText(board, text=s[1], org=(120,270), fontFace=FONT, fontScale=DOWN_scale, color=(pixel0,pixel0,pixel0), thickness=DOWN_thickness)
        elif len(s[1])==5:
            cv2.putText(board, text=s[1], org=(80,270), fontFace=FONT, fontScale=DOWN_scale, color=(pixel0,pixel0,pixel0), thickness=DOWN_thickness)

        cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = brightness(board)
        board = squeeze(board)
        # board = addmud(board)
        board = random_gaussian(board)
        board = motion_blur(board)
        board = addnoise(board)
        
        #board = arealight(board)
        board = mosaic(board)
        board = mosaic(board)
        board = PerspectiveTransform(board)
        
        # # view
        # cv2.imshow('', board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite('{}/{}night-{}.jpg'.format(path,str(i), p), board)


# if __name__ == '__main__':
#     FONT = cv2.FONT_ITALIC
#     BOARD = cv2.imread('plate3.png')
    
#     ### thickness
#     R_thickness = 12
#     L_thickness = 10
#     tk_inside = np.random.randint(5, 8) 
    
#     ### scale 
#     R_scale = 3.5
#     L_scale = 3
    
#     gen_plates(1000, 'imgs3')

if __name__ == '__main__':
    #FONT = cv2.FONT_ITALIC
    FONT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    BOARD = cv2.imread('plate2.png')
    
    ### thickness
    DOWN_thickness = 15
    UP_thickness = 10
    tk_inside = np.random.randint(7, 9) # 控制车牌上方数字边缘宽度
    
    ### scale 
    DOWN_scale = 5
    UP_scale = 3
    
    gen_plates(10, 'imgs2')














