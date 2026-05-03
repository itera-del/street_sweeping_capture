import cv2

import numpy as np
import numpy.random as random
import torch
from torchvision.transforms import Normalize as Normalize_th


class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  #(W, H)

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        segLabel_1 = sample.get('segLabel_1', None)
        segLabel_2 = sample.get('segLabel_2', None)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        if segLabel is not None:
            segLabel = cv2.resize(segLabel, self.size, interpolation=cv2.INTER_NEAREST)
        if segLabel_1 is not None:
            segLabel_1 = cv2.resize(segLabel_1, self.size, interpolation=cv2.INTER_NEAREST)
        if segLabel_2 is not None:
            segLabel_2 = cv2.resize(segLabel_2, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        _sample['segLabel_1'] = segLabel_1
        _sample['segLabel_2'] = segLabel_2
        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class RandomResize(Resize):
    """
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH)
    """
    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW+1) 
        h = np.random.randint(self.minH, self.maxH+1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        probability = np.random.rand(1,1)
        if probability < 0.6:
            return sample
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        degree = (u-0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        if segLabel is not None:
            segLabel = cv2.warpAffine(segLabel, R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta


class Light(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        probability = np.random.rand(1,1)
        if probability < 0.5:
            return sample
        img = sample.get('img')
        alpha = float(random.randint(5,15)/10.0)
        beta = random.randint(0,self.theta)
        s=random.randint(0,1)
        if s==0:
            s=-1
        img = np.uint8(np.clip((alpha * img + s*beta), 0, 255))
        _sample = sample.copy()
        _sample['img'] = img
        return _sample

    def reset_theta(self, theta):
        self.theta = theta



class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        img = sample.get('img')

        img = self.transform(img)

        _sample = sample.copy()
        _sample['img'] = img
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        segLabel_1 = sample.get('segLabel_1', None)
        segLabel_2 = sample.get('segLabel_2', None)
        segLabel_2_cla = sample.get('segLabel_2_cla', None)
        segLabel_cla = sample.get('segLabel_cla', None)
        exist = sample.get('exist', None)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.
        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)
        if segLabel_1 is not None:
            segLabel_1 = torch.from_numpy(segLabel_1).type(torch.long)
        if segLabel_2 is not None:
            segLabel_2 = torch.from_numpy(segLabel_2).type(torch.long)
        if segLabel_2_cla is not None:
            segLabel_2_cla = torch.from_numpy(segLabel_2_cla).type(torch.long)
        if segLabel_cla is not None:
            segLabel_cla = torch.from_numpy(segLabel_cla).type(torch.long)
        if exist is not None:
            exist = torch.from_numpy(exist).type(torch.float32)  # BCEloss requires float tensor

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        _sample['segLabel_1'] = segLabel_1
        _sample['segLabel_2'] = segLabel_2
        _sample['segLabel_2_cla'] = segLabel_2_cla
        _sample['segLabel_cla'] = segLabel_cla
        _sample['exist'] = exist
        return _sample


class GaussNoise(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        probability = np.random.rand(1,1)
        if probability < 0.5:
            return sample
        img = sample.get('img')
        kernel_size = 2*np.random.randint(2,4)+1
        sigma = np.random.randint(1,3)
        img = cv2.GaussianBlur(img, (kernel_size,kernel_size), sigma)
        _sample = sample.copy()
        _sample['img'] = img
        return _sample

    def reset_theta(self, theta):
        self.theta = theta

class Expand(CustomTransform):
    def __init__(self, theta):
        self.mean = np.random.randint(0,255)
        self.theta=theta

    def __call__(self, sample):
        if random.randint(2):
            return sample
        image = sample.get('img')
        segLabel = sample.get('segLabel', None)
        #print(image.shape, segLabel.shape)
        height, width, depth = image.shape
        ratio = (float)(random.uniform(100, self.theta))/100
        left = random.uniform(0, int(width*ratio) - width)
        top = random.uniform(0, int(height*ratio) - height)
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        expand_label = np.zeros(
            (int(height*ratio), int(width*ratio)),
            dtype=segLabel.dtype)
        expand_label[:, :] = 0
        expand_label[int(top):int(top + height),
                     int(left):int(left + width)] = segLabel
        # expand_image = cv2.resize(expand_image, (width,height), interpolation=cv2.INTER_NEAREST)
        # expand_label = cv2.resize(expand_label, (width,height), interpolation=cv2.INTER_NEAREST)
        # print(expand_image.shape, expand_label.shape)
        image = expand_image
        segLabel = expand_label
        _sample = sample.copy()
        _sample['img'] = image
        _sample['segLabel'] = segLabel
        return _sample

class Flip_y(CustomTransform):
    def __init__(self):
        print("flip")

    def __call__(self, sample):
        probability = np.random.rand(1,1)
        if probability < 0.7:
            return sample
        image = sample.get('img')
        segLabel = sample.get('segLabel', None)
        #print(image.shape, segLabel.shape)
        image = cv2.flip(image,0,dst=None) 
        segLabel = cv2.flip(segLabel,0,dst=None) 
        _sample = sample.copy()
        _sample['img'] = image
        _sample['segLabel'] = segLabel
        return _sample


class Crop(CustomTransform):
    def __init__(self, theta):
        self.theta=theta

    def __call__(self, sample):
        image = sample.get('img')
        segLabel = sample.get('segLabel', None)
        height, width, depth = image.shape
        ratio = (float)(random.uniform(self.theta, 90))/100
        new_width = int(width*ratio)
        new_height = int(height*ratio)
        left = random.uniform(100, width - new_width)
        top = random.uniform(100, height - new_height)
        expand_image = np.zeros(
            (new_height, new_width, depth),
            dtype=image.dtype)
        expand_image= image[int(top):int(top + new_height),
                     int(left):int(left + new_width)] 
        expand_label = np.zeros(
            (new_height, new_width),
            dtype=segLabel.dtype)
        expand_label[:, :] =  segLabel[int(top):int(top + new_height),
                     int(left):int(left + new_width)] 
        image = expand_image
        segLabel = expand_label
        _sample = sample.copy()
        _sample['img'] = image
        _sample['segLabel'] = segLabel
        return _sample