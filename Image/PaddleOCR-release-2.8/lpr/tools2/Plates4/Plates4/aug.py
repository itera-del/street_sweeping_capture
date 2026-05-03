import cv2
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
    beta  = np.random.uniform(0.0, 10.0)
    blank = np.zeros(img.shape, img.dtype)
    img = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return img

def arealight(img):
    h, w, _ = img.shape
    light = cv2.imread('src_aug/light.png')
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
    mud = cv2.imread('src_aug/smu.jpg')
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

def do_aug(board):
    board = squeeze(board)
    board = addmud(board)
    board = random_gaussian(board)
    board = motion_blur(board)
    board = addnoise(board)
    board = brightness(board)
    # board = arealight(board)
    board = mosaic(board)
    board = mosaic(board)
    board = PerspectiveTransform(board)
    # pass
    return board