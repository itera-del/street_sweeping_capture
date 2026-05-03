import os
import cv2
import sys
import tqdm
import utils
import numpy
import torch
#from apex import amp
from glob import glob
from tqdm import tqdm 
import torch.distributed
from dataset import segdata, clsdata
import models.LaneNet as LaneNet
import models.NovaLaneNet as NovaLaneNet
from dataset.transforms import *
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# caffe_root = '/home/workspace/pfji/enet_prun/caffe-enet/'
# sys.path.insert(0, caffe_root + 'python')  
# import caffe
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
        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class BeltSegTrainingEngine():
    """ This is a custom engine for this training cycle """
    def __init__(self, imglistPath, batchSize, inputSize=[3,128,128], savepath='./output/hand/', cls = True):
        self.early_stop = 50
        self.best_loss = 5
        self.savepath = savepath
        self.batchSize = batchSize
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        if not os.path.exists(savepath + '/result/'):
            os.mkdir(savepath + '/result/')
        self.model_name = savepath+'BeltNet.pth'
        self.best_model_name = savepath +'BeltNet_best.pth'
        weight=torch.tensor([0.5, 1])
        testweight=torch.tensor([0.5, 1])
        #self.model = LaneNet.LANENet(num_classes=2).cuda()
        self.model = NovaLaneNet.LANENET_NOVT_TEST(num_classes=2).cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)
        self.criterion_focal = FocalLoss2d(gamma=2, weight=weight).to(device)
        self.criterion_test  = torch.nn.CrossEntropyLoss(weight=testweight).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005,betas=(0.5, 0.999))
        self.transform_test = Compose(Flip_y(),Resize(tuple([inputSize[1],inputSize[2]])),ToTensor())
        transform_train = Compose(Flip_y(),Light(30),GaussNoise(0),Resize(tuple([inputSize[1],inputSize[2]])),ToTensor())
        valdata = segdata.Platedataset(imglistPath,"val",self.transform_test)
        testdata = segdata.Platedataset(imglistPath,"test",self.transform_test)
        traindata = segdata.Platedataset(imglistPath,"train", transform_train)
        self.Valdataloader = DataLoader(valdata, batch_size=batchSize, shuffle=True, collate_fn=valdata.collate, num_workers=4)
        self.Testdataloader = DataLoader(testdata, batch_size=batchSize, shuffle=True, collate_fn=testdata.collate, num_workers=4)
        self.Traindataloader = DataLoader(traindata, batch_size=batchSize, shuffle=True, collate_fn=traindata.collate, num_workers=4)
    #   self.model, self.optimizer = amp.initialize(self.crnn, self.optimizer, opt_level='O1')  # 这里是字母O

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
        self.optimizer.load_state_dict(checkpoint_finetune['optim'])
        self.best_loss = checkpoint_finetune.get("best_val_loss", 1e6)

    def train(self, epoch):
        print("Train Epoch: {}".format(epoch))
        self.model.train()
        train_loss = 0
        train_loss_seg = 0
        train_loss_exist = 0
        torch.set_grad_enabled(True)
        for batch_idx, sample in tqdm(enumerate(self.Traindataloader),total = len(self.Traindataloader)):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)
            self.optimizer.zero_grad()
            seg_pred= self.model(img, segLabel)
            loss = self.criterion(seg_pred,segLabel)
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            iter_idx = epoch * len(self.Traindataloader) + batch_idx
            train_loss = loss.item()
            if iter_idx % 200==0:
                print("batch loss: {:.3f}".format(loss.item()),batch_idx,len(self.Traindataloader)/self.batchSize)
        if epoch % 1 == 0:
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_loss
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
            segLabel = sample['segLabel'].to(device)
            seg_pred= self.model(img, segLabel)
            loss = self.criterion(seg_pred,segLabel)
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
            val_loss += loss.item()
        print("val_loss:{}, best_loss :{}".format(val_loss,self.best_loss))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_loss
            }
            torch.save(save_dict, self.best_model_name)
            print("model is saved: {}".format(self.best_model_name))
        print("------------------------\n")

    def test(self, epoch):
        self.model.eval()
        num = 0
        torch.set_grad_enabled(False)
        for batch_idx, sample in tqdm(enumerate(self.Testdataloader),total = len(self.Testdataloader)):
            img_cpu = sample['img']
            img = img_cpu.to(device)
            segLabel = sample['segLabel'].to(device)
            seg_pred = self.model(img, segLabel)
            seg_pred = seg_pred.detach().cpu().numpy()
            for b in range(len(img_cpu)):
                num +=1
                curImg = img_cpu[b]
                curImg = curImg.permute(1,2,0) * 255
                curImg = curImg.byte()
                img = curImg.numpy()                
                lane_img = np.zeros_like(img)
                coord_mask = np.argmax(seg_pred[b], axis=0)
                lane_img[coord_mask == 1] = [0,255,0]
                #lane_img[coord_mask == 2] = [0,0,255]
                img_save = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
                cat_img  = np.concatenate([img, img_save], axis=1)
                cv2.imwrite(self.savepath+"/result/"+str(num) + ".jpg",cat_img) 

    def testcaffe(self):
        num = 0
        protofile = './tools/belt_seg.prototxt'
        modelFile = './tools/belt_seg.caffemodel'
        net = caffe.Net(protofile,modelFile, caffe.TEST)
        for batch_idx, sample in tqdm(enumerate(self.Testdataloader),total = len(self.Testdataloader)):
            img_cpu = sample['img']
            for b in range(len(img_cpu)):
                num +=1
                curImg = img_cpu[b]
                caffeimage = curImg.numpy()
                curImg = curImg.permute(1,2,0) * 255
                curImg = curImg.byte()
                img = curImg.numpy()
                net.blobs['data'].data[...] = caffeimage
                preds = net.forward()["prob"]
                #preds=net.blobs['prob'].data
                print (preds.shape)
                seg_pred = preds[0]                
                coord_mask = np.argmax(seg_pred, axis=0)
                lane_img = np.zeros_like(img)
                lane_img[coord_mask == 1] = [0,255,0]
                #lane_img[coord_mask == 2] = [0,0,255]
                img_save = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
                cv2.imwrite(self.savepath+"/result/caffe"+str(num) + ".jpg",img_save) 
    def test_video(self, videopath):
        self.model.eval()
        torch.set_grad_enabled(False)
        vid_list = glob(videopath + '/*.mp4') 
        for i, v in enumerate(vid_list):
            print('*** running {}/{}'.format(i+1, len(vid_list)))
            video_reader = cv2.VideoCapture(v)
            num, fps = int(video_reader.get(7)), int(video_reader.get(5))
            hei, wid = int(video_reader.get(4)), int(video_reader.get(3))
            video_writer = cv2.VideoWriter(str(i)+".mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (wid, hei))
            for ii in tqdm(range(num)):
                (flag, image) = video_reader.read()
                mask = np.zeros_like(image)
                imageCrop = image[110:570,170:630,:]
                maskCrop = mask[110:570,170:630,:]
                x = self.transform_test({'img': imageCrop})['img']
                #x = self.transform_to_net({'img': imageCrop})['img']
                x.unsqueeze_(0)
                x=x.cuda()
                seg_pred = self.model(x)
                seg_pred = seg_pred.detach().cpu().numpy()
                coord_mask = np.argmax(seg_pred[0], axis=0 )
                coord_mask = np.array(coord_mask,dtype=np.uint8)
                coord_mask = cv2.resize(coord_mask, dsize=(460, 460), interpolation=cv2.INTER_NEAREST)
                #print(coord_mask.shape)
                maskCrop[coord_mask == 1] = [0,255,0]
                #maskCrop[coord_mask == 2] = [0,0,255]
                #maskCrop[coord_mask == 3] = [255,0,0]
                #maskCrop[coord_mask == 4] = [255,0,255]
                image = cv2.addWeighted(image, 1, mask, 1, 0)
                #cv2.rectangle(image, (300,0), (1500,1080), (255,255,255), 3)
                video_writer.write(image)
            video_writer.release()
            video_reader.release()

    def __call__A(self):
        Iteration = 0
        patiance = 0
        print(self.model)
        manualSeed=10
        best_accuracy=0
        numpy.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0   # replace all nan/inf in gradients to zero
        self.model.register_backward_hook(backward_hook)
        try:
            self.LoadPretrainModel(self.model_name)
            print('load pretrain model successfully!')
        except:
            print('train from the starting!')
        while Iteration < 400:
            Iteration+=1
            self.train(Iteration)
            self.val(Iteration)
            if Iteration % 5 == 0:
                self.test(Iteration)
    
            
    def __call__B(self):
        print(self.model)
        for idx, sample in tqdm(enumerate(self.Traindataloader),total = len(self.Traindataloader)):
            imgBatch = sample['img']
            label = sample['segLabel']
            for i in range(imgBatch.size(0)):
                curImg = imgBatch[i]
                curImg = curImg.permute(1,2,0) * 255
                curImg = curImg.byte()
                img = curImg.numpy()
                cv2.imwrite(str(i)+".jpg",img)

    def __call__(self):
        self.LoadPretrainModel(self.model_name)
        print('load pretrain model successfully!')
        self.test_video("./input/dataset_hand/video/")

class BeltClsEngine():
    """ This is a custom engine for this training cycle """
    def __init__(self, imglistPath, batchSize, inputSize=[3,128,128], savepath='./output/belt/', cls = True):
        self.early_stop = 50
        self.best_loss = 2
        self.best_acc = 0.5
        self.savepath = savepath
        self.batchSize = batchSize
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        if not os.path.exists(savepath + '/result/'):
            os.mkdir(savepath + '/result/')
        self.model_name = savepath+'BeltClsNet.pth'
        self.best_model_name = savepath +'BeltClsNet_best.pth'
        self.model = LaneNet.BeltNet(num_classes=2).cuda()
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.criterion_focal = FocalLoss2d(gamma=2).to(device)

        valdata = clsdata.HardSampleLoader(imglistPath + "/list/val.txt", tuple([inputSize[1],inputSize[2]]), False)
        testdata = clsdata.HardSampleLoader(imglistPath + "/list/test.txt", tuple([inputSize[1],inputSize[2]]), False)
        traindata = clsdata.HardSampleLoader(imglistPath + "/list/train.txt", tuple([inputSize[1],inputSize[2]]), True)
        self.Valdataloader = DataLoader(valdata, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Testdataloader = DataLoader(testdata, batch_size=batchSize, shuffle=True, num_workers=4)
        self.Traindataloader = DataLoader(traindata, batch_size=batchSize, shuffle=True, num_workers=4)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005,betas=(0.5, 0.999))
    #   self.model, self.optimizer = amp.initialize(self.crnn, self.optimizer, opt_level='O1')  # 这里是字母O

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
        self.optimizer.load_state_dict(checkpoint_finetune['optim'])
        self.best_loss = checkpoint_finetune.get("best_val_loss", 1e6)

    def train(self, epoch):
        print("Train Epoch: {}".format(epoch))
        self.model.train()
        torch.set_grad_enabled(True)
        for batch_idx, sample in tqdm(enumerate(self.Traindataloader),total = len(self.Traindataloader)):
            img = sample['img'].to(device)
            segLabel = sample['label'].to(device)
            self.optimizer.zero_grad()
            ce = self.model(img)
            ce = torch.squeeze(ce, dim=-1)
            label = torch.unsqueeze(segLabel, dim=1)
            loss = self.criterion(ce, label) 
            if isinstance(self.model, torch.nn.DataParallel):
                loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            iter_idx = epoch * len(self.Traindataloader) + batch_idx
            train_loss = loss.item()
            if iter_idx % 200==0:
                print("batch loss: {:.3f}".format(loss.item()),batch_idx,len(self.Traindataloader)/self.batchSize)
        if epoch % 1 == 0:
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_loss
            }
            torch.save(save_dict, self.model_name)
            print("model is saved: {}".format(self.model_name))
        print("----------------------------\n")

    def val(self, epoch):
        print("Val Epoch: {}".format(epoch))
        self.model.eval()
        val_loss = 0
        accuracy_ce = 0
        torch.set_grad_enabled(False)
        for batch_idx, sample in tqdm(enumerate(self.Valdataloader),total = len(self.Valdataloader)):
            img = sample['img'].to(device)
            label = sample['label'].to(device)
            self.optimizer.zero_grad()
            ce = self.model(img)
            ce  = torch.argmax(ce, dim=1)
            ce  = ce.cpu().numpy()
            for i in range(self.batchSize):
                c = ce[i, 0]
                b = label[i, 0].cpu()
                if c == b:
                    accuracy_ce += 1 
        accuracy_ce = accuracy_ce/len(valid_file_list)        
        print('epoch: {}/{}, valid acc(CE/FC) : {:.4f}'.format(epoch+1,  accuracy_ce))
        if accuracy_ce > self.best_acc:
            self.best_acc = accuracy_ce
            save_dict = {
                "net": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": self.best_loss
            }
            torch.save(save_dict, self.best_model_name)
            print("model is saved: {}".format(self.best_model_name))
        print("------------------------\n")

    # def test(self, epoch):
    #     self.model.eval()
    #     num = 0
    #     torch.set_grad_enabled(False)
    #     for batch_idx, sample in tqdm(enumerate(self.Testdataloader),total = len(self.Testdataloader)):
    #         img_cpu = sample['img']
    #         img = img_cpu.to(device)
    #         segLabel = sample['segLabel'].to(device)
    #         seg_pred, loss = self.model(img, segLabel)
    #         seg_pred = seg_pred.detach().cpu().numpy()
    #         for b in range(len(img_cpu)):
    #             num +=1
    #             curImg = img_cpu[b]
    #             curImg = curImg.permute(1,2,0) * 255
    #             curImg = curImg.byte()
    #             img = curImg.numpy()                
    #             lane_img = np.zeros_like(img)
    #             coord_mask = np.argmax(seg_pred[b], axis=0)
    #             lane_img[coord_mask == 1] = [0,255,0]
    #             lane_img[coord_mask == 2] = [0,0,255]
    #             img_save = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    #             cat_img  = np.concatenate([img, img_save], axis=1)
    #             cv2.imwrite(self.savepath+"/result/"+str(num) + ".jpg",cat_img) 




    def __call__A(self):
        Iteration = 0
        patiance = 0
        print(self.model)
        manualSeed=10
        best_accuracy=0
        numpy.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0   # replace all nan/inf in gradients to zero
        self.model.register_backward_hook(backward_hook)
        try:
            self.LoadPretrainModel(self.model_name)
            print('load pretrain model successfully!')
        except:
            print('train from the starting!')
        while Iteration < 200:
            Iteration+=1
            self.train(Iteration)
            self.val(Iteration)
            if Iteration % 30 == 0:
                self.test(Iteration)

    def __call__(self):
        self.LoadPretrainModel(self.model_name)
        print('load pretrain model successfully!')
        self.test_video("./input/dataset_handseg/video/")

    # def __call__(self):
    #     print(self.model)
    #     img = np.ones([1,3,128,128])
    #     img = img.astype(np.float32)
    #     input_var = Variable(torch.FloatTensor(torch.from_numpy(img)).cuda())
    #     self.model.eval()
    #     _=self.model(input_var)
    #     model_dict = self.model.state_dict()

    #     for idx, (k, v) in enumerate(model_dict.items()):
    #         print(idx, k)
    #     for idx, sample in tqdm(enumerate(self.Traindataloader),total = len(self.Traindataloader)):
    #         imgBatch = sample['img']
    #         label = sample['segLabel']
    #         for i in range(imgBatch.size(0)):
    #             curImg = imgBatch[i]
    #             curImg = curImg.permute(1,2,0) * 255
    #             curImg = curImg.byte()
    #             img = curImg.numpy()
    #             cv2.imwrite(str(i)+".jpg",img)


