import os
import cv2
import sys
import tqdm
import importlib
import numpy
import torch
from tqdm import tqdm 
import torch.distributed
from dataset import segdata
from loss.rmi_loss import RMILoss
from loss.focal_loss import SegFocalLoss
import models.NovaLaneNet as NovaLaneNet
from dataset.transforms import *
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')


class CitySegColorClasTrainingEngine():

    def __init__(self, img_list_path, batch_size, seg_city_dict_name='dataset_zd_dict', seg_color_dict_name='dataset_zd_dict_color', inputSize=[3,128,64], save_path='./output/seg_city/', lmdb_bool=False, lmdb_path=""):

        # dataset_zd_dict
        self.dataset_city_dict = importlib.import_module('script.lpr.dataset.dataset_zd_shate.dataset_dict.' + seg_city_dict_name) 
        self.dataset_color_dict = importlib.import_module('script.lpr.dataset.dataset_zd_shate.dataset_dict.' + seg_color_dict_name) 

        self.seg_city_labels = self.dataset_city_dict.class_seg_label
        self.seg_color_labels = self.dataset_color_dict.class_seg_label

        self.save_path = save_path
        self.batch_size = batch_size
        self.best_val_loss = 1000.0
        self.model_name = save_path + 'LaneNetNova2Head.pth'
        self.best_model_name = save_path + 'LaneNetNova2Head_best.pth'

        # mkdir
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # model
        self.model = NovaLaneNet.LANENET_NOVT_HEAD_SEG_CLA(num_classes_1_head=len(self.seg_city_labels), num_classes_2_head=len(self.seg_color_labels)).cuda()

        # criterion
        weight_city = torch.tensor(self.dataset_city_dict.class_seg_weight)
        self.criterion_city = torch.nn.CrossEntropyLoss(weight=weight_city).to(device)
        # self.criterion_rmi_city = RMILoss(num_classes=len(self.seg_city_labels)).to(device)
        
        weight_color = torch.tensor(self.dataset_color_dict.class_seg_weight)
        self.criterion_color = torch.nn.CrossEntropyLoss(weight=weight_color).to(device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, betas=(0.5, 0.999))

        # data
        transform_test = Compose(Resize(tuple([inputSize[1], inputSize[2]])), ToTensor())
        transform_train = Compose(Resize(tuple([inputSize[1], inputSize[2]])), ToTensor())
        valdata = segdata.Platedataset_HEAD_SEG_CLA(img_list_path, "val", transform_test, lmdb_bool=lmdb_bool, lmdb_path=lmdb_path)
        testdata = segdata.Platedataset_HEAD_SEG_CLA(img_list_path, "test", transform_test, lmdb_bool=lmdb_bool, lmdb_path=lmdb_path)
        traindata = segdata.Platedataset_HEAD_SEG_CLA(img_list_path, "trainval", transform_train, lmdb_bool=lmdb_bool, lmdb_path=lmdb_path)
        self.Valdataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=valdata.collate, num_workers=8)
        self.Testdataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=testdata.collate, num_workers=8)
        self.Traindataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=traindata.collate, num_workers=8)


    def LoadPretrainModel(self, path):
        checkpoint_finetune=torch.load(path)
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint_finetune['net']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(model_dict) 


    def train(self, epoch):
        print("Train Epoch: {}".format(epoch))
        self.model.train()
        torch.set_grad_enabled(True)

        for batch_idx, sample in tqdm(enumerate(self.Traindataloader), total=len(self.Traindataloader)):
            img = sample['img'].to(device)
            seg_label_1 = sample['segLabel_1'].to(device)
            seg_label_2 = sample['segLabel_2_cla'].to(device)

            self.optimizer.zero_grad()
            seg_pred_1, seg_pred_2 = self.model(img)

            loss_city = self.criterion_city(seg_pred_1, seg_label_1)
            # loss_city += self.criterion_rmi_city(seg_pred_1, seg_label_1)
            loss_color = self.criterion_color(seg_pred_2, seg_label_2) * 10
            loss = loss_city + loss_color

            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
                loss_city = loss_city.sum()
                loss_color = loss_color.sum()
            
            loss.backward()
            self.optimizer.step()

            iter_idx = epoch * len(self.Traindataloader) + batch_idx
            if iter_idx % 200==0:
                print("batch loss: {:.3f}, loss_city: {:.3f}, loss_color: {:.3f}".format(loss.item(), loss_city.item(), loss_color.item()), batch_idx, len(self.Traindataloader)/self.batch_size)

        if epoch % 1 == 0:
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss
            }
            torch.save(save_dict, self.model_name)
            print("model is saved: {}".format(self.model_name))
        print("------------------------\n")


    def val(self, epoch):

        if epoch % 10 != 0:
            return

        print("Val Epoch: {}".format(epoch))
        self.model.eval()
        val_loss = 0
        val_city_loss = 0
        val_color_loss = 0
        torch.set_grad_enabled(False)
        for batch_idx, sample in tqdm(enumerate(self.Valdataloader),total = len(self.Valdataloader)):
            img = sample['img'].to(device)
            seg_label_1 = sample['segLabel_1'].to(device)
            seg_label_2 = sample['segLabel_2_cla'].to(device)

            seg_pred_1, seg_pred_2 = self.model(img)

            loss_city = self.criterion_city(seg_pred_1, seg_label_1)
            loss_color = self.criterion_color(seg_pred_2, seg_label_2) * 10
            loss = loss_city + loss_color
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
                loss_city = loss_city.sum()
                loss_color = loss_color.sum()
            val_loss += loss.item()
            val_city_loss += loss_city.item()
            val_color_loss += loss_color.item()
        print("val_loss:{:.3f}, val_city_loss:{:.3f}, val_color_loss:{:.3f}, best_val_loss :{:.3f}".format(val_loss, val_city_loss, val_color_loss, self.best_val_loss))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss
            }
            torch.save(save_dict, self.best_model_name)
            print("model is saved: {}".format(self.best_model_name))
        print("------------------------\n")


    def test(self, epoch):
        
        if epoch % 50 != 0:
            return

        # mkdir
        if not os.path.exists(self.save_path + "/result/"):
            os.mkdir(self.save_path + "/result/")
        
        self.model.eval()
        num = 0
        torch.set_grad_enabled(False)
        for batch_idx, sample in tqdm(enumerate(self.Testdataloader), total=len(self.Testdataloader)):
            img_cpu = sample['img']
            img = img_cpu.to(device)

            seg_pred_1, seg_pred_2 = self.model(img)
            seg_pred_1 = seg_pred_1.detach().cpu().numpy()
            seg_pred_2 = seg_pred_2.detach().cpu().numpy()

            for idx in range(len(img_cpu)):
                num += 1

                # img
                img = img_cpu[idx]
                img = img.permute(1, 2, 0) * 255
                img = img.byte()
                img = img.numpy()

                # mask
                seg_mask_1 = np.argmax(seg_pred_1[idx], axis=0)
                seg_make_2 = np.argmax(seg_pred_2[idx], axis=0)
                mask_img = np.zeros(img.shape, dtype='int64')

                for chn_idx in range(len(self.seg_city_labels)):

                    if chn_idx == 0:
                        continue
                    
                    mask_img[seg_mask_1 == chn_idx] += self.dataset_city_dict.id_2_mask_color_dict[chn_idx]

                for chn_idx in range(len(self.seg_color_labels)):
                    
                    if chn_idx == 0:
                        continue
                    
                    if seg_make_2.max() == chn_idx:
                        mask_img += self.dataset_color_dict.id_2_mask_color_dict[chn_idx]

                mask_img = mask_img.astype(img.dtype)
                img_save = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=1.0, gamma=0.)
                cv2.imwrite(self.save_path+"/result/"+str(num) + ".jpg", img_save) 


    def __call__(self):

        # init 
        Iteration = 0
        manual_seed = 10
        numpy.random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        # print model
        print(self.model)

        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0   # replace all nan/inf in gradients to zero
        self.model.register_backward_hook(backward_hook)

        # load model
        try:
            self.LoadPretrainModel(self.model_name)
            print('load pretrain model successfully!')
        except:
            print('train from the starting!')

        # while Iteration < 500:
        while Iteration < 300:
        # while Iteration < 100:
            Iteration += 1
            self.train(Iteration)
            self.val(Iteration)
            self.test(Iteration)