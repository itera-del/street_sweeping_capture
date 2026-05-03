import math
import cv2
import numpy as np
import random


class RMRecAug(object):
    def __init__(self,
                 motion_blur_prob=0.2,
                 blur_prob=0.4,
                 expand_prob=0.4,
                 light_aug_prob=0.4,
                 reverse_prob=0.3,
                 **kwargs):
        self.motion_blur_prob = motion_blur_prob
        self.blur_prob = blur_prob
        self.expand_prob = expand_prob
        self.light_aug_prob = light_aug_prob
        self.reverse_prob = reverse_prob

    def __call__(self, data):
        img = data['image']

        probability = np.random.rand(1,1)
        if probability < self.motion_blur_prob:
            img = motion_blur(img) 

        probability = np.random.rand(1,1)
        if probability < self.blur_prob:
            img = blur(img)

        probability = np.random.rand(1,1)
        if probability < self.expand_prob:
            img = expand(img)

        probability = np.random.rand(1,1)
        if probability < 0.4:
            img = change_light(img)

        probability = np.random.rand(1,1)
        if probability < self.reverse_prob:
            img = 255 - img

        # # test
        # h, w = img.shape[0], img.shape[1]
        # img_path = "/yuanhuan/model/image/lpr/test/img_{}_{}.jpg".format(h, w)
        # cv2.imwrite(img_path, img)

        data['image'] = img
        return data


def motion_blur(img):

    degree = np.random.randint(3, 7)
    angle  = np.random.randint(0, 180)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred_img = cv2.filter2D(img, -1, motion_blur_kernel)
    
    cv2.normalize(blurred_img, blurred_img, 0, 255, cv2.NORM_MINMAX)
    blurred_img = np.array(blurred_img, dtype=np.uint8)

    return blurred_img


def blur(img):

    kernel_size = 2 * np.random.randint(2, 4) + 1
    sigma = np.random.randint(1, 3)

    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    return img


def expand(img):

    h, w, c = img.shape

    mean = random.uniform(0, 255)
    ratio = (float)(random.uniform(100, 125)) / 100
    left = random.uniform(0, int(w * ratio) - w)
    top = random.uniform(0, int(h * ratio) - h)

    expand_img = np.zeros((int(h * ratio), int(w * ratio), c), dtype=img.dtype)

    expand_img[:, :, :] = mean
    expand_img[int(top):int(top + h), int(left):int(left + w), :] = img

    return expand_img


def change_light(img):

    alpha = float(random.randint(7, 13) / 10.0)
    beta = random.randint(10, 20)
    s = random.randint(0, 1)

    if s == 0:
        s = -1

    img = np.uint8(np.clip((alpha * img + s * beta), 0, 255))

    return img


class RMRecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 padding=False,
                 bool_random_resize=False,
                 bool_white=False,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding
        self.bool_random_resize = bool_random_resize
        self.bool_white = bool_white

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_rm(img, 
                                                        self.image_shape,
                                                        self.padding,
                                                        False,
                                                        self.bool_white)
        else:
            norm_img, valid_ratio = resize_norm_img_rm(img, 
                                                        self.image_shape,
                                                        self.padding,
                                                        self.bool_random_resize,
                                                        self.bool_white)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


def resize_norm_img_rm(img,
                    image_shape,
                    padding=False,
                    bool_random_resize=False,
                    bool_white=False):
    
    img_reshape = img[: img.shape[0]//4*4, : img.shape[1]//4*4, :]

    h, w = img_reshape.shape[0], img_reshape.shape[1]
    imgC, imgH, imgW = image_shape

    if not padding:

        if imgC == 1:
            if bool_random_resize:
                resized_image = random_resize(img_reshape, (imgW, imgH))
            else:
                resized_image = cv2.resize(img_reshape, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            if bool_random_resize:
                resized_image = random_resize(img_reshape, (imgW, imgH))
            else:
                resized_image = cv2.resize(img_reshape, (imgW, imgH))

        # # test
        # img_path = "/yuanhuan/model/image/lpr/test/img_{}_{}.jpg".format(h, w)
        # cv2.imwrite(img_path, resized_image)

        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255

        valid_ratio = 1.0

        return resized_image, valid_ratio

    else:

        max_wh_ratio = imgW * 1.0 / imgH
        ratio = w * 1.0 / h

        # pad
        if ratio < max_wh_ratio:
            to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
            if (to_imgW - w < 32):
                to_imgW = w + 32

            if bool_white:
                pad_img = np.ones((h, to_imgW, 3), dtype=np.uint8)
                pad_img *= 255
                pad_img[:, 0:w, :] = img_reshape  
                valid_ratio = min(1.0, float(w / to_imgW))
            else:
                pad_img = np.zeros((h, to_imgW, 3), dtype=np.uint8)
                pad_img[:, 0:w, :] = img_reshape  
                valid_ratio = min(1.0, float(w / to_imgW))
            
        else:
            pad_img = img_reshape
            valid_ratio = 1.0

        if imgC == 1:
            if bool_random_resize:
                resized_image = random_resize(pad_img, (imgW, imgH))
            else:
                resized_image = cv2.resize(pad_img, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            if bool_random_resize:
                resized_image = random_resize(pad_img, (imgW, imgH))
            else:
                resized_image = cv2.resize(pad_img, (imgW, imgH))
        
        # # test
        # img_path = "/yuanhuan/model/image/lpr/test/img_{}_{}_{}.jpg".format(h, w, valid_ratio)
        # cv2.imwrite(img_path, resized_image)

        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255      

        return resized_image, valid_ratio


def random_resize(image, size):

    def bgr2nv21(bgr):
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        height = bgr.shape[0]
        width = bgr.shape[1]

        u = i420[height: height + height // 4, :]
        u = u.reshape((1, height // 4 * width))
        v = i420[height + height // 4: height + height // 2, :]
        v = v.reshape((1, height // 4 * width))
        uv = np.zeros((1, height // 4 * width * 2))
        uv[:, 0::2] = v
        uv[:, 1::2] = u
        uv = uv.reshape((height // 2, width))
        nv21 = np.zeros((height + height // 2, width))
        nv21[0:height, :] = i420[0:height, :]
        nv21[height::, :] = uv
        return nv21

    rand_id = random.randint(0,5)
    if(rand_id==0):
        image=cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    elif(rand_id==1):
        image=cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    elif(rand_id==2):
        yuv=cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        yuv=cv2.resize(yuv,(size[0], int(size[1]*1.5)),interpolation=cv2.INTER_LINEAR)
        image=cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420) 
    elif(rand_id==3):
        yuv=cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        yuv=cv2.resize(yuv,(size[0], int(size[1]*1.5)),interpolation=cv2.INTER_NEAREST)
        image=cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)           
    elif(rand_id==4):
        yuv=bgr2nv21(image)
        yuv=yuv.astype(np.uint8)
        yuv=cv2.resize(yuv,(size[0],int(size[1]*1.5)),interpolation=cv2.INTER_LINEAR)
        image=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR_NV21)   
    else:
        yuv=bgr2nv21(image)
        yuv=yuv.astype(np.uint8)
        yuv=cv2.resize(yuv,(size[0],int(size[1]*1.5)),interpolation=cv2.INTER_NEAREST)
        image=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR_NV21)      
    return image