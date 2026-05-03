#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ Sept 25th 2019 xnlin    
# 
import numpy as np
import cv2
import sys
import random
import os

def filp_image_x(img,box):
    x1,y1,x2,y2=box
    xImg = cv2.flip(img,1,dst=None) 
    xImg_x1 = 1 - x1
    xImg_x2 = 1 - x2
    xImge_label = [xImg_x1,y1,xImg_x2,y2]
    return xImg,xImge_label
def filp_image_y(img,x1,y1,x2,y2):
    yImg = cv2.flip(img,0,dst=None) 
    yImg_y1 = 1 - y1
    yImg_y2 = 1 - y2
    yImge_label = [x1,yImg_y1,x2,yImg_y2]    
    return yImg,yImge_label
def filp_image_xy(img,x1,y1,x2,y2):
    xyImg = cv2.flip(img,-1,dst=None)
    xyImg_y1 = 1 - y1
    xyImg_y2 = 1 - y2
    xyImg_x1 = 1 - x1
    xyImg_x2 = 1 - x2    
    xyImge_label = [xyImg_x1,xyImg_y1,xyImg_x2,xyImg_y2]    
    return xyImg,xyImge_label
def Gaussian_nosie(img):
    kernel_size = 2*np.random.randint(2,4)+1
    sigma = np.random.randint(1,3)
    #print(kernel_size, sigma)
    img = cv2.GaussianBlur(img, (kernel_size,kernel_size), sigma)
    return img
def solt_nosie(img):
    percetage = random.randint(1,5)*0.01
    NoiseNum=int(percetage*img.shape[0]*img.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,img.shape[0]-1)
        randY=np.random.random_integers(0,img.shape[1]-1)
        img[randX,randY]=np.random.random_integers(0,255)
    return img
def color_change(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height,width,3),np.uint8)
    boffset=random.randint(-30,30)
    goffset=random.randint(-30,30)
    roffset=random.randint(-30,30)
    for i in range(0,height):    
        for j in range(0,width):       
            (b,g,r) = img[i,j]
            dst[i,j] = (min(255,max(0,b-boffset)),min(255,max(g-goffset,0)),min(255,max(0,r-roffset)))
    return dst
def change_light(img):
    alpha = float(random.randint(7,13)/10.0)
    beta = random.randint(10,20)
    s=random.randint(0,1)
    if s==0:
        s=-1
    img = np.uint8(np.clip((alpha * img + s*beta), 0, 255))
    return img


def rotate(img):
    u = np.random.uniform()
    degree = (u-0.5) * 8
    R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
    img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return img

def expand(image):
    if len(image.shape) == 2:
        height, width = image.shape
        mean = random.uniform(0, 255)
        ratio = (float)(random.uniform(100, 125)) / 100
        left = random.uniform(0, int(width*ratio) - width)
        top = random.uniform(0, int(height*ratio) - height)
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio)),
            dtype=image.dtype)
        expand_image[:, :] = mean
        expand_image[int(top):int(top + height),
                    int(left):int(left + width)] = image
        image = expand_image
    elif len(image.shape) == 3:
        height, width, chn = image.shape
        mean = random.uniform(0, 255)
        ratio = (float)(random.uniform(100, 125)) / 100
        left = random.uniform(0, int(width*ratio) - width)
        top = random.uniform(0, int(height*ratio) - height)
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), chn),
            dtype=image.dtype)
        expand_image[:, :, :] = mean
        expand_image[int(top):int(top + height),
                    int(left):int(left + width), :] = image
        image = expand_image

    return image

def aug_box(img,box):
    probability = np.random.rand(1,1)
    label=box
    if probability < 0.2:
        #img,label = filp_image_x(img,box)
        pass
    probability = np.random.rand(1,1)
    if probability < 0.3:
        img = Gaussian_nosie(img)
    probability = np.random.rand(1,1)
    if probability < 0.3:
        img = color_change(img)
    probability = np.random.rand(1,1)
    if probability < 0.3:
        img = change_light(img)
    return img,label

def motion_blur(image):
    degree = np.random.randint(3, 7)
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

def imgaug(img):
    probability = np.random.rand(1,1)
    if probability < 0.3:
        img[:,:] = 255 - img
    probability = np.random.rand(1,1)
    if probability < 0.2:
        img = motion_blur(img) 
    probability = np.random.rand(1,1)
    if probability < 0.3:
        img = rotate(img)
    probability = np.random.rand(1,1)
    if probability < 0.4:
        img = expand(img)
    probability = np.random.rand(1,1)
    if probability < 0.4:
        img = Gaussian_nosie(img)
        img = change_light(img)
    return img


def imgaug_cityseg(img):
    probability = np.random.rand(1,1)
    if probability < 0.2:
        img = motion_blur(img) 
    probability = np.random.rand(1,1)
    if probability < 0.4:
        img = Gaussian_nosie(img)
        img = change_light(img)
    return img


def imgaugtest(img):
    probability = np.random.rand(1,1)
    if probability < 0.2:
        img = cv2.flip(img,0,dst=None) 
    probability = np.random.rand(1,1)
    if probability < 0.4:
        img = change_light(img)
    return img