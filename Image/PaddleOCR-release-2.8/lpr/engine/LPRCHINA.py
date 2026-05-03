# -*- coding: utf-8 -*-
import os
import cv2
import sys
import tqdm
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
# caffe_root = '/home/workspace/xnlin/MobileNet-SSD-Focal-loss/'
# sys.path.insert(0, caffe_root + 'python')  
# import caffe

class TrainingEngine():
    """ This is a custom engine for this training cycle """
    def __init__(self, imglistPath, imgtestlistPath, batchSize, inputSize=[1,64,300], savepath="./output/china/double"):
        ad_labels = ["-", "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']
        self.early_stop = 500
        self.best_accuracy = 0.1
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.converter=dubaiplate.Str2Labelchina(ad_labels)
        #self.crnn = crnnv2.CRNNv4(nclass=len(ad_labels)).cuda()
        self.crnn = crnnv2.CRNNv3(nclass=len(ad_labels)).cuda()
        self.criterion = torch.nn.CTCLoss( reduction='sum').cuda()
        traindata = dubaiplate.DatasetChinaPlate(imglistPath,inputSize=[1,64,300],istrain=1)
        testdata_b = dubaiplate.DatasetChinaPlate(imgtestlistPath[0]+"/B/",inputSize=[1,64,300],istrain=0)
        testdata_n = dubaiplate.DatasetChinaPlate(imgtestlistPath[0]+"/N/",inputSize=[1,64,300],istrain=0)
        testdata_y1 = dubaiplate.DatasetChinaPlate(imgtestlistPath[0]+"/Y1/",inputSize=[1,64,300],istrain=0)
        testdata_y2 = dubaiplate.DatasetChinaPlate(imgtestlistPath[0]+"/Y2/",inputSize=[1,64,300],istrain=0)
        self.Traindataloader = DataLoader(traindata, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Valdataloader_b = DataLoader(testdata_b, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Valdataloader_n = DataLoader(testdata_n, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Valdataloader_y1 = DataLoader(testdata_y1, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Valdataloader_y2 = DataLoader(testdata_y2, batch_size=batchSize, shuffle=True, num_workers=4)
        self.optimizer = torch.optim.Adam(self.crnn.parameters(), lr=0.005,betas=(0.5, 0.999))
        #self.crnn, self.optimizer = amp.initialize(self.crnn, self.optimizer, opt_level='O1')  # 这里是字母O

    def LoadPretrainModel(self, path):
        pretrained_dict=torch.load(path)
        model_dict = self.crnn.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.crnn.load_state_dict(model_dict)

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

    def val(self, iteration, Valdataloader, max_i=1000):
        print('Start val')
        all_num=0
        n_correct = 0
        self.crnn.eval()
        torch.set_grad_enabled(False)
        loss_avg = utils.averager()
        for i_batch, (image, label) in tqdm(enumerate(Valdataloader),total=len(Valdataloader)):
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

    # def test_caffe(self, iteration, max_i=1000):
    #     print('Start val')
    #     all_num=0
    #     n_correct = 0
    #     n_error = 0
    #     protofile = './tools/cnn_turkey.prototxt'
    #     modelFile = './tools/turkey4.caffemodel'
    #     net = caffe.Net(protofile,modelFile, caffe.TEST)
    #     for i_batch, (image, label) in tqdm(enumerate(self.Valdataloader),total=len(self.Valdataloader)):
    #         image = image.to(device)
    #         preds = self.crnn(image)
    #         batch_size = image.size(0)
    #         text, length = self.converter.encode(label)
    #         preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #         cost = self.criterion(preds, text, preds_size, length) / batch_size
    #         loss_avg.add(cost)
    #         _, preds = preds.max(2)
    #         preds = preds.transpose(1, 0).contiguous().view(-1)
    #         sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
    #         for pred, curImg,target in zip(sim_preds, image, label):
    #             all_num=all_num+1
    #             if pred == target:
    #                 n_correct += 1
    #             else:
    #                 n_error +=1
    #                 curImg = curImg.permute(1,2,0) * 255
    #                 curImg = curImg.byte()
    #                 img = curImg.cpu().numpy()
    #                 cv2.imwrite(self.savepath+"/result/"+str(n_error)+"_"+target+"pred_"+pred+".jpg",img)                    
    #         if i_batch == max_i:
    #             break
    #     raw_preds = self.converter.decode(preds.data, preds_size.data, raw=True)[:16]
    #     for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
    #         print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    #     print(n_correct)
    #     print(all_num)
    #     accuracy = n_correct / float(all_num)
    #     print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    #     return accuracy



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
        while Iteration < 200:
            Iteration+=1
            self.train(Iteration)
            print("*******************test blue plate *****************************")
            accuracy1 = self.val(Iteration, self.Valdataloader_b, max_i=len(self.Valdataloader_b))
            print("*******************test new plate *****************************")
            accuracy2 = self.val(Iteration, self.Valdataloader_n, max_i=len(self.Valdataloader_n))
            print("*******************test yellow1 plate *****************************")
            accuracy3 = self.val(Iteration, self.Valdataloader_y1, max_i=len(self.Valdataloader_y1))
            print("*******************test yellow2 plate *****************************")
            accuracy4 = self.val(Iteration, self.Valdataloader_y2, max_i=len(self.Valdataloader_y2))
            accuracy = accuracy2
            # accuracy = accuracy4
            if accuracy > self.best_accuracy:
                patiance = 0
                self.best_accuracy = accuracy
                #torch.save(self.crnn.state_dict(), '{0}/crnn.pth'.format(self.savepath))
                torch.save(self.crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(self.savepath, Iteration, accuracy))
                torch.save(self.crnn.state_dict(), '{0}/crnn_best.pth'.format(self.savepath))
            else:
                patiance=patiance+1
            if patiance>self.early_stop:
                break                
            print("is best accuracy: {0}, best accurary is : {1}".format(accuracy > best_accuracy, self.best_accuracy))
            # if Iteration % 30 == 0:
            #     accuracy = self.test(Iteration,max_i=len(self.Valdataloader))
        print("is best accuracy: {0}, best accurary is : {1}".format(accuracy > best_accuracy, self.best_accuracy))
            
    def __call__B(self):
        for idx, data in tqdm(enumerate(self.Traindataloader),total = len(self.Traindataloader)):
            imgBatch = data[0]
            label = data[1]
            for i in range(imgBatch.size(0)):
                curImg = imgBatch[i]
                curImg = curImg.permute(1,2,0) * 255
                curImg = curImg.byte()
                img = curImg.numpy()
                cv2.imwrite(str(i)+".jpg",img)

