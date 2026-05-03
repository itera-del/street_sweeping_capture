# -*- coding: utf-8 -*-

import os
import cv2
import sys
import tqdm
import importlib
import utils
import numpy
import torch
#from apex import amp
from tqdm import tqdm 
import torch.distributed
import models.crnnv2 as crnnv2
from dataset import dubaiplate
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')
# sys.path.insert(0, '/yuanhuan')


class TrainingEngine():
    """ This is a custom engine for this training cycle """
    def __init__(self, imglistPath, batchSize, gray_bool=True, padding_bool=True, early_stop=300, data_dict_name="lpr.script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict", savepath='./output/dubai/all_time_lstm/'):

        # init
        self.gray_bool = gray_bool
        self.padding_bool = padding_bool
        # ocr_labels = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        #       'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','#']
        # ocr_labels = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0','#']
        
        # dataset_zd_dict
        self.dataset_dict = importlib.import_module(data_dict_name) 
        ocr_labels = self.dataset_dict.kind_num_labels

        self.early_stop = early_stop
        self.best_accuracy = 0.1
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.converter=dubaiplate.Str2Label(ocr_labels)

        # # 统计计算量和参数量
        # self.crnn = crnnv2.CRNNv3(nclass=len(ocr_labels))

        # from torchstat import stat
        # stat(self.crnn, (1, 64, 256))

        # def count_parameters(model):
        #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        # params_num = count_parameters(self.crnn)
        # print("par.:  {}K".format(params_num/1000.0))

        if self.gray_bool:
            self.crnn = crnnv2.CRNNv3(nclass=len(ocr_labels)).cuda()
        else:
            self.crnn = crnnv2.CRNNv3_BGR(nclass=len(ocr_labels)).cuda()
        # self.crnn = crnnv2.CRNNv4(nclass=len(ocr_labels)).cuda()
        # self.crnn = crnnv2.CRNNv5(nclass=len(ocr_labels)).cuda()
        # self.crnn = crnnv2.CRNNv6(nclass=len(ocr_labels)).cuda()
        
        self.criterion = torch.nn.CTCLoss( reduction='sum').cuda()
        traindata = dubaiplate.DatasetPlateSet(imglistPath, "train", gray_bool = self.gray_bool, padding_bool = self.padding_bool)
        testdata = dubaiplate.DatasetPlateSet(imglistPath, "val", gray_bool = self.gray_bool, padding_bool = self.padding_bool)
        self.Traindataloader = DataLoader(traindata, batch_size=batchSize, shuffle=True, num_workers=8)
        self.Valdataloader = DataLoader(testdata, batch_size=batchSize, shuffle=True, num_workers=8)
        self.optimizer = torch.optim.Adam(self.crnn.parameters(), lr=0.005,betas=(0.5, 0.999))
        # self.optimizer = torch.optim.Adam(self.crnn.parameters(), lr=0.00001,betas=(0.5, 0.999))
        #self.crnn, self.optimizer = amp.initialize(self.crnn, self.optimizer, opt_level='O1')  # 这里是字母O

    def LoadPretrainModel(self, path):
        pretrained_dict=torch.load(path)
        model_dict = self.crnn.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.crnn.load_state_dict(model_dict, strict=False)

    def train(self, iteration):
        self.crnn.train()
        torch.set_grad_enabled(True)
        loss_avg = utils.averager()
        for i_batch, (image, label) in tqdm(enumerate(self.Traindataloader),total=len(self.Traindataloader)):
            image = image.to(device)
            preds = self.crnn(image)
            batch_size = image.size(0)
            text, length = self.converter.encode(label)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = self.criterion(preds, text, preds_size, length) / batch_size
            self.crnn.zero_grad()
            cost.backward()
            # with amp.scale_loss(cost, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            self.optimizer.step()
            loss_avg.add(cost)
            if (i_batch+1) % 300 == 0:
                print(' Loss: %f' %(loss_avg.val()))
                loss_avg.reset() 

    def val(self, iteration, max_i=1000):
        print('Start val')
        all_num=0
        n_correct = 0
        self.crnn.eval()
        torch.set_grad_enabled(False)
        loss_avg = utils.averager()
        for i_batch, (image, label) in tqdm(enumerate(self.Valdataloader),total=len(self.Valdataloader)):
            image = image.to(device)
            preds = self.crnn(image)
            batch_size = image.size(0)
            text, length = self.converter.encode(label)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = self.criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, label):
                all_num=all_num+1
                # if pred[0] =='#':
                #     pred=pred[1:-1]
                if pred == target:
                    n_correct += 1
            if i_batch == max_i:
                break
        raw_preds = self.converter.decode(preds.data, preds_size.data, raw=True)[:10]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
            # if pred[0] =='#':
            #     pred=pred[1:-1]
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        print(n_correct)
        print(all_num)
        accuracy = n_correct / float(all_num)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        return accuracy

    def test(self, iteration, max_i=1000):
        print('Start val')
        all_num=0
        n_correct = 0
        n_error = 0
        self.crnn.eval()
        torch.set_grad_enabled(False)
        loss_avg = utils.averager()
        for i_batch, (image, label) in tqdm(enumerate(self.Valdataloader),total=len(self.Valdataloader)):
            image = image.to(device)
            preds = self.crnn(image)
            batch_size = image.size(0)
            text, length = self.converter.encode(label)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = self.criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for pred, curImg,target in zip(sim_preds, image, label):
                all_num=all_num+1
                if pred == target:
                    n_correct += 1
                else:
                    n_error +=1
                    curImg = curImg.permute(1,2,0) * 255
                    curImg = curImg.byte()
                    img = curImg.cpu().numpy()
                    cv2.imwrite(self.savepath+"/result/"+str(n_error)+"_"+target+"pred_"+pred+".jpg",img)                    
            if i_batch == max_i:
                break
        raw_preds = self.converter.decode(preds.data, preds_size.data, raw=True)[:16]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        print(n_correct)
        print(all_num)
        accuracy = n_correct / float(all_num)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        return accuracy


    def __call__(self):
        Iteration = 0
        patiance = 0
        manualSeed=10
        best_accuracy=0
        numpy.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0   # replace all nan/inf in gradients to zero
        self.crnn.register_backward_hook(backward_hook)
        try:
            self.LoadPretrainModel('{0}/crnn_best.pth'.format(self.savepath))
            print('load pretrain model successfully!')
        except:
            print('train from the starting!')
        # while Iteration < 500:
        while Iteration < 300:
        # while Iteration < 100:
            Iteration+=1
            self.train(Iteration)
            accuracy = self.val(Iteration, max_i=len(self.Valdataloader))
            if accuracy > self.best_accuracy:
                patiance = 0
                self.best_accuracy = accuracy
                torch.save(self.crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(self.savepath, Iteration, accuracy))
                torch.save(self.crnn.state_dict(), '{0}/crnn_best.pth'.format(self.savepath))
            else:
                patiance = patiance + 1

            if patiance > self.early_stop:
                break                
            # print("Iteration: {0}/{1}, is best accuracy: {2}, best accurary is : {3}".format(Iteration, 500, accuracy > best_accuracy,self.best_accuracy))
            print("Iteration: {0}/{1}, is best accuracy: {2}, best accurary is : {3}".format(Iteration, 300, accuracy > best_accuracy,self.best_accuracy))
        print("is best accuracy: {0}, best accurary is : {1}".format(accuracy > best_accuracy, self.best_accuracy))