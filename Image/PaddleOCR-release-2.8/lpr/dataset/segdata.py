import cv2
import io
import lmdb
import numpy as np
import os
import torch
import random
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.transforms import *
from dataset import aug

class Platedataset(Dataset):
    def __init__(self, path, image_set, transforms=None, mask_type="png"):
        super(Platedataset, self).__init__()
        assert image_set in ('train', 'val', 'test', 'trainval'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.mask_type = mask_type
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print("image_set: {}, plate nums: {}".format(image_set, len(self.img_list)))

    def createIndex(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                # # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_list.append(l[1])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path = line.split(".jpg ")[1]
                self.img_list.append(jpg_path)
                self.segLabel_list.append(seg_path)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_list))
        random.shuffle(c)
        self.img_list, self.segLabel_list = zip(*c)

    def createIndex_test(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                # # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_list.append(l[1])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path = line.split(".jpg ")[1]
                self.img_list.append(jpg_path)
                self.segLabel_list.append(seg_path)

    def __getitem__(self, idx):        
        try :
            # print(self.img_list[idx])
            img = cv2.imread(self.img_list[idx])
            # print(self.segLabel_list[idx])
            if self.mask_type == "png":
                segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            elif self.mask_type == "pkl":
                f = open(self.segLabel_list[idx], 'rb')
                segLabel = pickle.load(f)
                f.close()
        except:
            print(self.img_list[idx])
            return self[idx+1]

        if(self.image_set == 'train'):
            img = aug.imgaug_cityseg(img)

        sample = {'img': img,
                  'segLabel': segLabel,
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'img_name': [x['img_name'] for x in batch]}

        return samples


class Platedataset_2Head(Dataset):
    def __init__(self, path, image_set, transforms=None, mask_type="png"):
        super(Platedataset_2Head, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.mask_type = mask_type
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print(len(self.img_list))

    def createIndex(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_1_list = []
        self.segLabel_2_list = []
        with open(listfile) as f:
            for line in f:
                # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_1_list.append(l[1])
                # self.segLabel_2_list.append(l[2])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path_1 = line.split(".jpg ")[1].split(".png")[0] + '.png'
                seg_path_2 = line.split(".jpg ")[1].split(".png ")[1]
                if not seg_path_2.endswith('.png'):
                    seg_path_2 += '.png'
                self.img_list.append(jpg_path)
                self.segLabel_1_list.append(seg_path_1)
                self.segLabel_2_list.append(seg_path_2)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_1_list, self.segLabel_2_list))
        random.shuffle(c)
        self.img_list, self.segLabel_1_list, self.segLabel_2_list = zip(*c)

    def createIndex_test(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_1_list = []
        self.segLabel_2_list = []
        with open(listfile) as f:
            for line in f:
                # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_1_list.append(l[1])
                # self.segLabel_2_list.append(l[2])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path_1 = line.split(".jpg ")[1].split(".png")[0] + '.png'
                seg_path_2 = line.split(".jpg ")[1].split(".png ")[1]
                if not seg_path_2.endswith('.png'):
                    seg_path_2 += '.png'
                self.img_list.append(jpg_path)
                self.segLabel_1_list.append(seg_path_1)
                self.segLabel_2_list.append(seg_path_2)

    def __getitem__(self, idx):        
        try :
            # print(self.img_list[idx])
            img = cv2.imread(self.img_list[idx])
            # print(self.segLabel_1_list[idx])
            # print(self.segLabel_2_list[idx])
            if self.mask_type == "png":
                segLabel_1 = cv2.imread(self.segLabel_1_list[idx])[:, :, 0]
                segLabel_2 = cv2.imread(self.segLabel_2_list[idx])[:, :, 0]
            elif self.mask_type == "pkl":
                f = open(self.segLabel_1_list[idx], 'rb')
                segLabel_1 = pickle.load(f)
                f.close()

                f = open(self.segLabel_2_list[idx], 'rb')
                segLabel_2 = pickle.load(f)
                f.close()
        except:
            print(self.img_list[idx])
            return self[idx+1]

        if(self.image_set == 'train'):
            img = aug.imgaug_cityseg(img)

        sample = {'img': img,
                  'segLabel_1': segLabel_1,
                  'segLabel_2': segLabel_2,
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel_1'] is None:
            segLabel_1 = None
        elif isinstance(batch[0]['segLabel_1'], torch.Tensor):
            segLabel_1 = torch.stack([b['segLabel_1'] for b in batch])
        else:
            segLabel_1 = [b['segLabel_1'] for b in batch]

        if batch[0]['segLabel_2'] is None:
            segLabel_2 = None
        elif isinstance(batch[0]['segLabel_2'], torch.Tensor):
            segLabel_2 = torch.stack([b['segLabel_2'] for b in batch])
        else:
            segLabel_2 = [b['segLabel_2'] for b in batch]

        samples = {'img': img,
                  'segLabel_1': segLabel_1,
                  'segLabel_2': segLabel_2,
                  'img_name': [x['img_name'] for x in batch]}

        return samples

class Platedataset_HEAD_SEG_CLA(Dataset):
    def __init__(self, path, image_set, transforms=None, mask_type="png", lmdb_bool=False, lmdb_path=""):
        super(Platedataset_HEAD_SEG_CLA, self).__init__()
        assert image_set in ('train', 'val', 'test', 'trainval'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.mask_type = mask_type
        self.lmdb_bool = lmdb_bool
        self.lmdb_path = lmdb_path
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print("image_set: {}, plate nums: {}".format(image_set, len(self.img_list)))

        if self.lmdb_bool:
            self.load_lmdb_env_dict()

    def load_lmdb_env(self, lmdb_path):
        lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return lmdb_env

    def read_lmdb(self, env, key):
        # with env.begin(write=False) as txn:
        #     buf = txn.get(key.encode())        
        buf = self.lmdb_env_to_txn[env].get(key.encode())
        byte_data = np.fromstring(buf, dtype=np.uint8)
        load_bytes = io.BytesIO(byte_data)
        img_loads = np.load(load_bytes)
        return img_loads

    def load_lmdb_env_dict(self):
        self.lmdb_env_dict = {}
        self.lmdb_env_to_txn = {}

        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        with open(listfile) as f:
            for line in f:
                jpg_path = line.split(".jpg")[0] + '.jpg'
                env_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(jpg_path))))
                if env_name not in self.lmdb_env_dict:
                    print("lmdb env name: {}".format(env_name))
                    lmdb_path = os.path.join(self.lmdb_path, env_name, "{}.lmdb".format(self.image_set))
                    self.lmdb_env_dict[env_name] = self.load_lmdb_env(lmdb_path)
                    self.lmdb_env_to_txn[env_name] = self.lmdb_env_dict[env_name].begin(write=False)

    def createIndex(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_1_list = []
        self.segLabel_2_list = []
        with open(listfile) as f:
            for line in f:
                # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_1_list.append(l[1])
                # self.segLabel_2_list.append(l[2])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path_1 = line.split(".jpg ")[1].split(".png")[0] + '.png'
                seg_path_2 = line.split(".jpg ")[1].split(".png ")[1]
                if not seg_path_2.endswith('.png'):
                    seg_path_2 += '.png'
                self.img_list.append(jpg_path)
                self.segLabel_1_list.append(seg_path_1)
                self.segLabel_2_list.append(seg_path_2)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_1_list, self.segLabel_2_list))
        random.shuffle(c)
        self.img_list, self.segLabel_1_list, self.segLabel_2_list = zip(*c)

    def createIndex_test(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_1_list = []
        self.segLabel_2_list = []
        with open(listfile) as f:
            for line in f:
                # old style: 不适配文件名带 “ ” 字符
                # line = line.strip()
                # l = line.split(" ")
                # self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                # self.segLabel_1_list.append(l[1])
                # self.segLabel_2_list.append(l[2])

                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path_1 = line.split(".jpg ")[1].split(".png")[0] + '.png'
                seg_path_2 = line.split(".jpg ")[1].split(".png ")[1]
                if not seg_path_2.endswith('.png'):
                    seg_path_2 += '.png'
                self.img_list.append(jpg_path)
                self.segLabel_1_list.append(seg_path_1)
                self.segLabel_2_list.append(seg_path_2)

    def __getitem__(self, idx):        
        try :
            if not self.lmdb_bool:
                # print(self.img_list[idx])
                img = cv2.imread(self.img_list[idx])
                # print(self.segLabel_1_list[idx])
                # print(self.segLabel_2_list[idx])
                if self.mask_type == "png":
                    segLabel_1 = cv2.imread(self.segLabel_1_list[idx])[:, :, 0]
                    segLabel_2 = cv2.imread(self.segLabel_2_list[idx])[:, :, 0]
                elif self.mask_type == "pkl":
                    f = open(self.segLabel_1_list[idx], 'rb')
                    segLabel_1 = pickle.load(f)
                    f.close()

                    f = open(self.segLabel_2_list[idx], 'rb')
                    segLabel_2 = pickle.load(f)
                    f.close()
            else:
                jpg_path = self.img_list[idx]
                seg_path_1 = self.segLabel_1_list[idx]
                seg_path_2 = self.segLabel_2_list[idx]
                env_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(jpg_path))))
                img = self.read_lmdb(env_name, jpg_path)
                segLabel_1 = self.read_lmdb(env_name, seg_path_1)[:, :, 0]
                segLabel_2 = self.read_lmdb(env_name, seg_path_2)[:, :, 0]

            # segLabel_2_cla = np.zeros((2, 4))
            # if segLabel_2.max() > 0 :
            #     segLabel_2_cla[:, :] = segLabel_2.max()

            # max-pool
            segLabel_2_cla = np.zeros((1))
            if segLabel_2.max() > 0 :
                segLabel_2_cla[0] = segLabel_2.max()

        except:
            print(self.img_list[idx])
            return self[idx+1]

        if(self.image_set == 'train'):
            img = aug.imgaug_cityseg(img)

        sample = {'img': img,
                  'segLabel_1': segLabel_1,
                  'segLabel_2_cla': segLabel_2_cla,
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel_1'] is None:
            segLabel_1 = None
        elif isinstance(batch[0]['segLabel_1'], torch.Tensor):
            segLabel_1 = torch.stack([b['segLabel_1'] for b in batch])
        else:
            segLabel_1 = [b['segLabel_1'] for b in batch]

        if batch[0]['segLabel_2_cla'] is None:
            segLabel_2_cla = None
        elif isinstance(batch[0]['segLabel_2_cla'], torch.Tensor):
            segLabel_2_cla = torch.stack([b['segLabel_2_cla'] for b in batch])
        else:
            segLabel_2_cla = [b['segLabel_2_cla'] for b in batch]

        samples = {'img': img,
                  'segLabel_1': segLabel_1,
                  'segLabel_2_cla': segLabel_2_cla,
                  'img_name': [x['img_name'] for x in batch]}

        return samples


class Platedataset_HEAD_CLA(Dataset):
    def __init__(self, path, image_set, transforms=None, mask_type="png"):
        super(Platedataset_HEAD_CLA, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.mask_type = mask_type
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print("image_set: {}, plate nums: {}".format(image_set, len(self.img_list)))

    def createIndex(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path = line.split(".jpg ")[1].split(".png")[0] + '.png'
                self.img_list.append(jpg_path)
                self.segLabel_list.append(seg_path)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_list))
        random.shuffle(c)
        self.img_list, self.segLabel_list = zip(*c)

    def createIndex_test(self):
        listfile = self.data_dir_path + "/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                # new style
                line = line.strip()
                jpg_path = line.split(".jpg")[0] + '.jpg'
                seg_path = line.split(".jpg ")[1].split(".png")[0] + '.png'
                self.img_list.append(jpg_path)
                self.segLabel_list.append(seg_path)

    def __getitem__(self, idx):        
        try :
            # print(self.img_list[idx])
            img = cv2.imread(self.img_list[idx])
            # print(self.segLabel_list[idx])
            if self.mask_type == "png":
                segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            elif self.mask_type == "pkl":
                f = open(self.segLabel_list[idx], 'rb')
                segLabel = pickle.load(f)
                f.close()
            
            # segLabel_cla = np.zeros((1, 2))
            segLabel_cla = np.zeros((2, 4))
            if segLabel.max() > 0 :
                segLabel_cla[:, :] = segLabel.max()

        except:
            print(self.img_list[idx])
            return self[idx+1]

        if(self.image_set == 'train'):
            img = aug.imgaug_cityseg(img)

        sample = {'img': img,
                  'segLabel_cla': segLabel_cla,
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel_cla'] is None:
            segLabel_cla = None
        elif isinstance(batch[0]['segLabel_cla'], torch.Tensor):
            segLabel_cla = torch.stack([b['segLabel_cla'] for b in batch])
        else:
            segLabel_cla = [b['segLabel_cla'] for b in batch]

        samples = {'img': img,
                  'segLabel_cla': segLabel_cla,
                  'img_name': [x['img_name'] for x in batch]}

        return samples
    

def RandomBrightness(img):
    alpha = np.random.uniform(0.5, 1.0)
    beta  = np.random.uniform(0.0, 10.0)
    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def CropArea(img1, img2):
    xmin = 920
    ymin = 0
    xmax = 1920
    ymax = 800
    return img1[ymin:ymax, xmin:xmax, :], img2[ymin:ymax, xmin:xmax]

def RandomCropArea(img1, img2):
    xmin = 920
    ymin = 0
    xmax = 1920
    ymax = 800
    lenx = xmax - xmin
    leny = ymax - ymin
    offset = np.random.randint(-280,280)
    lenx = lenx + offset
    leny = leny + offset
    xpoint = np.random.randint(0, 1920-lenx)
    ypoint = np.random.randint(0, 1080-leny)
    return img1[ypoint:ypoint+leny, xpoint:xpoint+lenx, :], img2[ypoint:ypoint+leny, xpoint:xpoint+lenx]


class BsdSegData(torch.utils.data.Dataset):
    '''
    provide: \n 
    src \n 
    mask1 (onehot, `c, h, w`) \n 
    mask2 (argmax, `1, h, w`)
    '''
    def __init__(self, path, image_set, transforms=None):
        super(BsdSegData, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print(len(self.img_list))

    def createIndex(self):
        #listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        listfile = self.data_dir_path +  "/list/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        #self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(l[1])
                #self.exist_list.append([int(x) for x in l[2:]])
        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_list))
        random.shuffle(c)
        self.img_list, self.segLabel_list = zip(*c)

    def createIndex_test(self):
        #listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        listfile = self.data_dir_path + "/list/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                #self.img_list.append(os.path.join(self.data_dir_path, line[1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(l[1])

    def __getitem__(self, idx):       
        try :
            img = cv2.imread(self.img_list[idx])
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            if self.image_set != 'test':
                img, segLabel = RandomCropArea(img, segLabel)
            else:
                img, segLabel = CropArea(img, segLabel) 
        except:
            print(self.img_list[idx])
            return self[idx+1]
        probability = np.random.rand(1,1)
        if probability < 0.3:
            imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img[:,:,0]=imggray
            img[:,:,1]=imggray
            img[:,:,2]=imggray
        # else probability < 0.3:
        #     imggray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sample = {'img': img,
                  'segLabel': segLabel,
                  'img_name': self.img_list[idx]}
        #cv2.imwrite("./output/bsd/result/src"+str(idx) + ".jpg",img) 
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'img_name': [x['img_name'] for x in batch]}

        return samples


class BsdSegRFData(torch.utils.data.Dataset):
    '''
    provide: \n 
    src \n 
    mask1 (onehot, `c, h, w`) \n 
    mask2 (argmax, `1, h, w`)
    '''
    def __init__(self, path, image_set, transforms=None):
        super(BsdSegRFData, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()
        print(len(self.img_list))

    def createIndex(self):
        #listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        listfile = self.data_dir_path +  "/list/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        #self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(l[1])
                #self.exist_list.append([int(x) for x in l[2:]])
        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self.img_list, self.segLabel_list))
        random.shuffle(c)
        self.img_list, self.segLabel_list = zip(*c)

    def createIndex_test(self):
        #listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        listfile = self.data_dir_path + "/list/" + "{}.txt".format(self.image_set)
        self.img_list = []
        self.segLabel_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                #self.img_list.append(os.path.join(self.data_dir_path, line[1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.img_list.append(l[0])   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(l[1])

    def __getitem__(self, idx):       
        try :
            img = cv2.imread(self.img_list[idx])
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
        except:
            print(self.img_list[idx])
            return self[idx+1]
        if self.image_set != 'test':
            probability = np.random.rand(1,1)
            if probability < 0.3:
                imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img[:,:,0]=imggray
                img[:,:,1]=imggray
                img[:,:,2]=imggray
            # elif probability < 0.6:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sample = {'img': img,
                  'segLabel': segLabel,
                  'img_name': self.img_list[idx]}
        #cv2.imwrite("./output/bsd/result/src"+str(idx) + ".jpg",img) 
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'img_name': [x['img_name'] for x in batch]}

        return samples


if __name__=="__main__":

    transform_train = Compose([256, 128], ToTensor())
    #train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #                          collate_fn=train_dataset.collate, num_workers=0)