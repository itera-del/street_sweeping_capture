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
import models.NovaLaneNet as NovaLaneNet
from dataset.transforms import *
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# caffe_root = '/home/workspace/pfji/enet_prun/caffe-enet/'
# sys.path.insert(0, caffe_root + 'python')  
# import caffe

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')

class FocalLoss2d(torch.nn.Module):
    '''
    input : b, c, w, h (outputs of the net) \n
    target: b, 1, w, h (label)
    '''
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt
        print(input.dim(),loss)
        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        #print(input.size())
        if input.dim()>2:
            input = F.log_softmax(input,dim=1)
            # input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            # input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            # input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        target = target.view(-1,1)
        logpt = input.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        
        if self.alpha is not None:
            self.alpha = Variable(torch.tensor(self.alpha))
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = (2.0*intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class CitySegTrainingEngine():
    """ This is a custom engine for this training cycle """
    def __init__(self, imglistPath, batch_size, seg_dict_name="script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict", inputSize=[3,128,64], save_path='./output/seg_city/'):

        # dataset_zd_dict
        self.dataset_dict = importlib.import_module(seg_dict_name) 

        self.seg_labels = self.dataset_dict.class_seg_label

        self.save_path = save_path
        self.batch_size = batch_size
        self.best_val_loss = 1000.0
        self.model_name = save_path + 'LaneNetNova.pth'
        self.best_model_name = save_path + 'LaneNetNova_best.pth'

        # mkdir
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # model
        self.model = NovaLaneNet.LANENET_NOVT_TEST(num_classes=len(self.seg_labels)).cuda()

        # criterion
        weight = torch.tensor(self.dataset_dict.class_seg_weight)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)
        self.criterion_rmi = RMILoss(num_classes=len(self.seg_labels)).to(device)
        # self.criterion_focal = FocalLoss(gamma=2, alpha=None, size_average=True).to(device)
        # self.criterion_focal = SigmoidFocalLoss(gamma=2, alpha=0.75).to(device)
        # self.criterion_test  = torch.nn.CrossEntropyLoss(weight=weight).to(device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005,betas=(0.5, 0.999))
        
        # data
        transform_test = Compose(Resize(tuple([inputSize[1], inputSize[2]])), ToTensor())
        transform_train = Compose(Resize(tuple([inputSize[1], inputSize[2]])), ToTensor())
        valdata = segdata.Platedataset(imglistPath, "val", transform_test)
        testdata = segdata.Platedataset(imglistPath, "test", transform_test)
        traindata = segdata.Platedataset(imglistPath, "trainval", transform_train)
        self.Valdataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=valdata.collate, num_workers=4)
        self.Testdataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=testdata.collate, num_workers=4)
        self.Traindataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=traindata.collate, num_workers=4)

    def LoadPretrainModel(self, path):
        checkpoint_finetune=torch.load(path)
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint_finetune['net']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
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
            seg_label = sample['segLabel'].to(device)

            self.optimizer.zero_grad()
            seg_pred = self.model(img)

            loss1 = self.criterion(seg_pred, seg_label)
            loss2 = self.criterion_rmi(seg_pred, seg_label)
            loss = loss2 + loss1
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
            
            loss.backward()
            self.optimizer.step()

            iter_idx = epoch * len(self.Traindataloader) + batch_idx
            if iter_idx % 200==0:
                print("batch loss: {:.3f}".format(loss.item()), batch_idx, len(self.Traindataloader)/self.batch_size)

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
        print("Val Epoch: {}".format(epoch))
        self.model.eval()
        val_loss = 0
        torch.set_grad_enabled(False)
        for batch_idx, sample in tqdm(enumerate(self.Valdataloader),total = len(self.Valdataloader)):
            img = sample['img'].to(device)
            seg_label = sample['segLabel'].to(device)

            seg_pred = self.model(img)

            loss = self.criterion(seg_pred, seg_label)
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
            val_loss += loss.item()
        print("val_loss:{}, best_val_loss :{}".format(val_loss, self.best_val_loss))
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

        if epoch % 10 != 0:
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
            seg_label = sample['segLabel'].to(device)

            seg_pred = self.model(img)
            seg_pred = seg_pred.detach().cpu().numpy()

            for idx in range(len(img_cpu)):
                num +=1

                # img
                curImg = img_cpu[idx]
                curImg = curImg.permute(1, 2, 0) * 255
                curImg = curImg.byte()
                img = curImg.numpy()                

                # mask
                seg_mask = np.argmax(seg_pred[idx], axis=0)
                mask_img = np.zeros(img.shape, dtype='int64')
                
                for chn_idx in range(len(self.seg_labels)):
                    
                    if chn_idx == 0:
                        continue

                    mask_img[seg_mask == chn_idx] += self.dataset_dict.id_2_mask_color_dict[chn_idx]

                mask_img = mask_img.astype(img.dtype)
                img_save = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=1.0, gamma=0.)
                cv2.imwrite(self.save_path+"/result/"+str(num) + ".jpg", img_save) 


    # def testcaffe(self):
    #     num = 0
    #     protofile = './tools/deploy_old.prototxt'
    #     modelFile = './tools/lanenet_new.caffemodel'
    #     net = caffe.Net(protofile,modelFile, caffe.TEST)
    #     for batch_idx, sample in tqdm(enumerate(self.Testdataloader), total = len(self.Testdataloader)):
    #         img_cpu = sample['img']
    #         for b in range(len(img_cpu)):
    #             num +=1
    #             curImg = img_cpu[b]
    #             caffeimage = curImg.numpy()
    #             curImg = curImg.permute(1, 2, 0) * 255
    #             curImg = curImg.byte()
    #             img = curImg.numpy()
    #             net.blobs['data'].data[...] = caffeimage
    #             preds = net.forward()["prob"]
    #             #preds=net.blobs['prob'].data
    #             print (preds.shape)
    #             seg_pred = preds[0]                
    #             seg_mask = np.argmax(seg_pred, axis=0)
    #             mask_img = np.zeros_like(img)
    #             mask_img[seg_mask == 1] = [0,255,0]
    #             mask_img[seg_mask == 2] = [0,0,255]
    #             mask_img[seg_mask == 3] = [255,0,0]
    #             mask_img[seg_mask == 4] = [255,0,255]
    #             mask_img[seg_mask == 5] = [0,255,255]
    #             mask_img[seg_mask == 6] = [255,255,0]
    #             mask_img[seg_mask == 9] = [0,0,0]
    #             img_save = cv2.addWeighted(src1=mask_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    #             cv2.imwrite(self.save_path+"/result/caffe"+str(num) + ".jpg",img_save) 


    def __call__(self):

        # init 
        Iteration = 0
        manualSeed=10
        numpy.random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # print model
        print(self.model)

        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0   # replace all nan/inf in gradients to zero
        self.model.register_backward_hook(backward_hook)

        try:
            self.LoadPretrainModel(self.model_name)
            print('load pretrain model successfully!')
        except:
            print('train from the starting!')

        # while Iteration < 100:
        while Iteration < 500:
            Iteration += 1
            self.train(Iteration)
            self.val(Iteration)
            self.test(Iteration)
