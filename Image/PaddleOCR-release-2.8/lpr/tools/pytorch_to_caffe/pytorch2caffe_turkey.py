#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ Oct 12th 2019 xnlin    
# 

import sys
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
import models.crnnv2 as crnnv2

# caffe_root = '/home/workspace/xnlin/MobileNet-SSD-Focal-loss/'
caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe

# model_path = "/mnt/huanyuan/model/image/lpr/bm/ocr_lxn/crnn_best_1020.pth"
# model_path = "/mnt/huanyuan/model/image/lpr/bm/ocr_lxn/crnn_best_1109.pth"
model_path = "/mnt/huanyuan/model/image/lpr/bm/ocr_lxn/crnn_best_1201.pth"
# model_path ="/home/workspace/xnlin/pytorch/visionlab/LPR/output/mexico/crnn_best.pth"
# model_path ="/home/workspace/xnlin/pytorch/chinese-ocr/expr_crnn_white_single3/crnn_best.pth"
# model_path = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_2_line_256_32_20230116/crnn_best.pth"

nameDict = {
    "cnn.0.0_0_layer.layers.0.weight" : "Convolution0,0",
    "cnn.0.0_0_layer.layers.1.weight" : "Scale0,0",
    "cnn.0.0_0_layer.layers.1.bias" : "Scale0,1",
    "cnn.0.0_0_layer.layers.1.running_mean" : "BatchNorm0,0",
    "cnn.0.0_0_layer.layers.1.running_var" : "BatchNorm0,1",
    "cnn.0.0_0_layer.layers.1.num_batches_tracked" : "BatchNorm0,2",

    "cnn.0.0_layer.layers.0.weight" : "Convolution1,0",
    "cnn.0.0_layer.layers.1.weight" : "Scale1,0",
    "cnn.0.0_layer.layers.1.bias" : "Scale1,1",
    "cnn.0.0_layer.layers.1.running_mean" : "BatchNorm1,0",
    "cnn.0.0_layer.layers.1.running_var" : "BatchNorm1,1",
    "cnn.0.0_layer.layers.1.num_batches_tracked" : "BatchNorm1,2",

    "cnn.0.1_layer.0.conv.0.weight" : "1_layer_0_Convolution1/dw,0",
    "cnn.0.1_layer.0.conv.1.weight" : "1_layer_0_Scale1,0",
    "cnn.0.1_layer.0.conv.1.bias" : "1_layer_0_Scale1,1",
    "cnn.0.1_layer.0.conv.1.running_mean" : "1_layer_0_BatchNorm1,0",
    "cnn.0.1_layer.0.conv.1.running_var" : "1_layer_0_BatchNorm1,1",
    "cnn.0.1_layer.0.conv.1.num_batches_tracked" : "1_layer_0_BatchNorm1,2",
    "cnn.0.1_layer.0.conv.3.weight" : "1_layer_0_Convolution2,0",
    "cnn.0.1_layer.0.conv.4.weight" : "1_layer_0_Scale2,0",
    "cnn.0.1_layer.0.conv.4.bias" : "1_layer_0_Scale2,1",
    "cnn.0.1_layer.0.conv.4.running_mean" : "1_layer_0_BatchNorm2,0",
    "cnn.0.1_layer.0.conv.4.running_var" : "1_layer_0_BatchNorm2,1",
    "cnn.0.1_layer.0.conv.4.num_batches_tracked" : "1_layer_0_BatchNorm2,2",

    "cnn.0.2_layer.0.conv.0.weight" : "2_layer_0_Convolution1/dw,0",
    "cnn.0.2_layer.0.conv.1.weight" : "2_layer_0_Scale1,0",
    "cnn.0.2_layer.0.conv.1.bias" : "2_layer_0_Scale1,1",
    "cnn.0.2_layer.0.conv.1.running_mean" : "2_layer_0_BatchNorm1,0",
    "cnn.0.2_layer.0.conv.1.running_var" : "2_layer_0_BatchNorm1,1",
    "cnn.0.2_layer.0.conv.1.num_batches_tracked" : "2_layer_0_BatchNorm1,2",
    "cnn.0.2_layer.0.conv.3.weight" : "2_layer_0_Convolution2,0",
    "cnn.0.2_layer.0.conv.4.weight" : "2_layer_0_Scale2,0",
    "cnn.0.2_layer.0.conv.4.bias" : "2_layer_0_Scale2,1",
    "cnn.0.2_layer.0.conv.4.running_mean" : "2_layer_0_BatchNorm2,0",
    "cnn.0.2_layer.0.conv.4.running_var" : "2_layer_0_BatchNorm2,1",
    "cnn.0.2_layer.0.conv.4.num_batches_tracked" : "2_layer_0_BatchNorm2,2",

    "cnn.0.2_layer.1.conv.0.weight" : "2_layer_1_Convolution1/dw,0",
    "cnn.0.2_layer.1.conv.1.weight" : "2_layer_1_Scale1,0",
    "cnn.0.2_layer.1.conv.1.bias" : "2_layer_1_Scale1,1",
    "cnn.0.2_layer.1.conv.1.running_mean" : "2_layer_1_BatchNorm1,0",
    "cnn.0.2_layer.1.conv.1.running_var" : "2_layer_1_BatchNorm1,1",
    "cnn.0.2_layer.1.conv.1.num_batches_tracked" : "2_layer_1_BatchNorm1,2",
    "cnn.0.2_layer.1.conv.3.weight" : "2_layer_1_Convolution2,0",
    "cnn.0.2_layer.1.conv.4.weight" : "2_layer_1_Scale2,0",
    "cnn.0.2_layer.1.conv.4.bias" : "2_layer_1_Scale2,1",
    "cnn.0.2_layer.1.conv.4.running_mean" : "2_layer_1_BatchNorm2,0",
    "cnn.0.2_layer.1.conv.4.running_var" : "2_layer_1_BatchNorm2,1",
    "cnn.0.2_layer.1.conv.4.num_batches_tracked" : "2_layer_1_BatchNorm2,2",


    "cnn.0.3_layer.0.conv.0.weight" : "3_layer_0_Convolution1,0",
    "cnn.0.3_layer.0.conv.1.weight" : "3_layer_0_Scale1,0",
    "cnn.0.3_layer.0.conv.1.bias" : "3_layer_0_Scale1,1",
    "cnn.0.3_layer.0.conv.1.running_mean" : "3_layer_0_BatchNorm1,0",
    "cnn.0.3_layer.0.conv.1.running_var" : "3_layer_0_BatchNorm1,1",
    "cnn.0.3_layer.0.conv.1.num_batches_tracked" : "3_layer_0_BatchNorm1,2",
    "cnn.0.3_layer.0.conv.3.weight" : "3_layer_0_Convolution2/dw,0",
    "cnn.0.3_layer.0.conv.4.weight" : "3_layer_0_Scale2,0",
    "cnn.0.3_layer.0.conv.4.bias" : "3_layer_0_Scale2,1",
    "cnn.0.3_layer.0.conv.4.running_mean" : "3_layer_0_BatchNorm2,0",
    "cnn.0.3_layer.0.conv.4.running_var" : "3_layer_0_BatchNorm2,1",
    "cnn.0.3_layer.0.conv.4.num_batches_tracked" : "3_layer_0_BatchNorm2,2",
    "cnn.0.3_layer.0.conv.6.weight" : "3_layer_0_Convolution3,0",
    "cnn.0.3_layer.0.conv.7.weight" : "3_layer_0_Scale3,0",
    "cnn.0.3_layer.0.conv.7.bias" : "3_layer_0_Scale3,1",
    "cnn.0.3_layer.0.conv.7.running_mean" : "3_layer_0_BatchNorm3,0",
    "cnn.0.3_layer.0.conv.7.running_var" : "3_layer_0_BatchNorm3,1",
    "cnn.0.3_layer.0.conv.7.num_batches_tracked" : "3_layer_0_BatchNorm3,2",

    "cnn.0.3_layer.1.conv.0.weight" : "3_layer_1_Convolution1,0",
    "cnn.0.3_layer.1.conv.1.weight" : "3_layer_1_Scale1,0",
    "cnn.0.3_layer.1.conv.1.bias" : "3_layer_1_Scale1,1",
    "cnn.0.3_layer.1.conv.1.running_mean" : "3_layer_1_BatchNorm1,0",
    "cnn.0.3_layer.1.conv.1.running_var" : "3_layer_1_BatchNorm1,1",
    "cnn.0.3_layer.1.conv.1.num_batches_tracked" : "3_layer_1_BatchNorm1,2",
    "cnn.0.3_layer.1.conv.3.weight" : "3_layer_1_Convolution2/dw,0",
    "cnn.0.3_layer.1.conv.4.weight" : "3_layer_1_Scale2,0",
    "cnn.0.3_layer.1.conv.4.bias" : "3_layer_1_Scale2,1",
    "cnn.0.3_layer.1.conv.4.running_mean" : "3_layer_1_BatchNorm2,0",
    "cnn.0.3_layer.1.conv.4.running_var" : "3_layer_1_BatchNorm2,1",
    "cnn.0.3_layer.1.conv.4.num_batches_tracked" : "3_layer_1_BatchNorm2,2",
    "cnn.0.3_layer.1.conv.6.weight" : "3_layer_1_Convolution3,0",
    "cnn.0.3_layer.1.conv.7.weight" : "3_layer_1_Scale3,0",
    "cnn.0.3_layer.1.conv.7.bias" : "3_layer_1_Scale3,1",
    "cnn.0.3_layer.1.conv.7.running_mean" : "3_layer_1_BatchNorm3,0",
    "cnn.0.3_layer.1.conv.7.running_var" : "3_layer_1_BatchNorm3,1",
    "cnn.0.3_layer.1.conv.7.num_batches_tracked" : "3_layer_1_BatchNorm3,2",


    "cnn.0.4_layer.0.conv.0.weight" : "4_layer_0_Convolution1,0",
    "cnn.0.4_layer.0.conv.1.weight" : "4_layer_0_Scale1,0",
    "cnn.0.4_layer.0.conv.1.bias" : "4_layer_0_Scale1,1",
    "cnn.0.4_layer.0.conv.1.running_mean" : "4_layer_0_BatchNorm1,0",
    "cnn.0.4_layer.0.conv.1.running_var" : "4_layer_0_BatchNorm1,1",
    "cnn.0.4_layer.0.conv.1.num_batches_tracked" : "4_layer_0_BatchNorm1,2",
    "cnn.0.4_layer.0.conv.3.weight" : "4_layer_0_Convolution2/dw,0",
    "cnn.0.4_layer.0.conv.4.weight" : "4_layer_0_Scale2,0",
    "cnn.0.4_layer.0.conv.4.bias" : "4_layer_0_Scale2,1",
    "cnn.0.4_layer.0.conv.4.running_mean" : "4_layer_0_BatchNorm2,0",
    "cnn.0.4_layer.0.conv.4.running_var" : "4_layer_0_BatchNorm2,1",
    "cnn.0.4_layer.0.conv.4.num_batches_tracked" : "4_layer_0_BatchNorm2,2",
    "cnn.0.4_layer.0.conv.6.weight" : "4_layer_0_Convolution3,0",
    "cnn.0.4_layer.0.conv.7.weight" : "4_layer_0_Scale3,0",
    "cnn.0.4_layer.0.conv.7.bias" : "4_layer_0_Scale3,1",
    "cnn.0.4_layer.0.conv.7.running_mean" : "4_layer_0_BatchNorm3,0",
    "cnn.0.4_layer.0.conv.7.running_var" : "4_layer_0_BatchNorm3,1",
    "cnn.0.4_layer.0.conv.7.num_batches_tracked" : "4_layer_0_BatchNorm3,2",

    "cnn.0.4_layer.1.conv.0.weight" : "4_layer_1_Convolution1,0",
    "cnn.0.4_layer.1.conv.1.weight" : "4_layer_1_Scale1,0",
    "cnn.0.4_layer.1.conv.1.bias" : "4_layer_1_Scale1,1",
    "cnn.0.4_layer.1.conv.1.running_mean" : "4_layer_1_BatchNorm1,0",
    "cnn.0.4_layer.1.conv.1.running_var" : "4_layer_1_BatchNorm1,1",
    "cnn.0.4_layer.1.conv.1.num_batches_tracked" : "4_layer_1_BatchNorm1,2",
    "cnn.0.4_layer.1.conv.3.weight" : "4_layer_1_Convolution2/dw,0",
    "cnn.0.4_layer.1.conv.4.weight" : "4_layer_1_Scale2,0",
    "cnn.0.4_layer.1.conv.4.bias" : "4_layer_1_Scale2,1",
    "cnn.0.4_layer.1.conv.4.running_mean" : "4_layer_1_BatchNorm2,0",
    "cnn.0.4_layer.1.conv.4.running_var" : "4_layer_1_BatchNorm2,1",
    "cnn.0.4_layer.1.conv.4.num_batches_tracked" : "4_layer_1_BatchNorm2,2",
    "cnn.0.4_layer.1.conv.6.weight" : "4_layer_1_Convolution3,0",
    "cnn.0.4_layer.1.conv.7.weight" : "4_layer_1_Scale3,0",
    "cnn.0.4_layer.1.conv.7.bias" : "4_layer_1_Scale3,1",
    "cnn.0.4_layer.1.conv.7.running_mean" : "4_layer_1_BatchNorm3,0",
    "cnn.0.4_layer.1.conv.7.running_var" : "4_layer_1_BatchNorm3,1",
    "cnn.0.4_layer.1.conv.7.num_batches_tracked" : "4_layer_1_BatchNorm3,2",

    "cnn.0.5_layer.0.conv.0.weight" : "5_layer_0_Convolution1/dw,0",
    "cnn.0.5_layer.0.conv.1.weight" : "5_layer_0_Scale1,0",
    "cnn.0.5_layer.0.conv.1.bias" : "5_layer_0_Scale1,1",
    "cnn.0.5_layer.0.conv.1.running_mean" : "5_layer_0_BatchNorm1,0",
    "cnn.0.5_layer.0.conv.1.running_var" : "5_layer_0_BatchNorm1,1",
    "cnn.0.5_layer.0.conv.1.num_batches_tracked" : "5_layer_0_BatchNorm1,2",
    "cnn.0.5_layer.0.conv.3.weight" : "5_layer_0_Convolution2,0",
    "cnn.0.5_layer.0.conv.4.weight" : "5_layer_0_Scale2,0",
    "cnn.0.5_layer.0.conv.4.bias" : "5_layer_0_Scale2,1",
    "cnn.0.5_layer.0.conv.4.running_mean" : "5_layer_0_BatchNorm2,0",
    "cnn.0.5_layer.0.conv.4.running_var" : "5_layer_0_BatchNorm2,1",
    "cnn.0.5_layer.0.conv.4.num_batches_tracked" : "5_layer_0_BatchNorm2,2",

    "cnn.0.5_layer.1.conv.0.weight" : "5_layer_1_Convolution1/dw,0",
    "cnn.0.5_layer.1.conv.1.weight" : "5_layer_1_Scale1,0",
    "cnn.0.5_layer.1.conv.1.bias" : "5_layer_1_Scale1,1",
    "cnn.0.5_layer.1.conv.1.running_mean" : "5_layer_1_BatchNorm1,0",
    "cnn.0.5_layer.1.conv.1.running_var" : "5_layer_1_BatchNorm1,1",
    "cnn.0.5_layer.1.conv.1.num_batches_tracked" : "5_layer_1_BatchNorm1,2",
    "cnn.0.5_layer.1.conv.3.weight" : "5_layer_1_Convolution2,0",
    "cnn.0.5_layer.1.conv.4.weight" : "5_layer_1_Scale2,0",
    "cnn.0.5_layer.1.conv.4.bias" : "5_layer_1_Scale2,1",
    "cnn.0.5_layer.1.conv.4.running_mean" : "5_layer_1_BatchNorm2,0",
    "cnn.0.5_layer.1.conv.4.running_var" : "5_layer_1_BatchNorm2,1",
    "cnn.0.5_layer.1.conv.4.num_batches_tracked" : "5_layer_1_BatchNorm2,2",

    "rnn.0.weight" : "Conv_r,0",
    "rnn.0.bias" : "Conv_r,1",
    "rnn.1.weight" : "Scale_r,0",
    "rnn.1.bias" : "Scale_r,1",
    "rnn.1.running_mean" : "BatchNorm_r,0",
    "rnn.1.running_var" : "BatchNorm_r,1",
    "rnn.1.num_batches_tracked" : "BatchNorm_r,2",

    "cnn_layer_new.weight" : "cnn_layer,0",
    "cnn_layer_new.bias" : "cnn_layer,1",

}
i=0
test=1
trans=1
if __name__ == '__main__':
    protofile = "/home/huanyuan/code/demo/Image/recognition2d/lpr/tools/prototxt/cnn_turkey.prototxt"
    # modelFile = "/mnt/huanyuan/model/image/lpr/bm/ocr_lxn/mexico_1109.caffemodel"
    modelFile = "/mnt/huanyuan/model/image/lpr/bm/ocr_lxn/mexico_1201_test.caffemodel"
    # modelFile = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_2_line_256_32_20230116/ocr_brazil_2_line_256_32_20230116.caffemodel"
    net = caffe.Net(protofile, caffe.TEST)
    caffeParams = net.params
    # print(len(caffeParams), type(caffeParams))
    # print(caffeParams)
    # print("OK")
    pytorchLayerNameList = list(nameDict.keys())
    caffeLayerNameList = list(nameDict.values())
    network=crnnv2.CRNNv4(38, 128)
    #network=crnnv2.CRNNv3(37, 128)
    network.load_state_dict(torch.load(model_path))
    network.eval()
    if trans:
        for param_tensor in network.state_dict():
            pytorchLayerName = param_tensor
            #print (param_tensor)
            #continue
            #print(i, param_tensor, nameDict[param_tensor])
            if pytorchLayerName not in pytorchLayerNameList:
                print(pytorchLayerName)
                #sys.exit()
                continue
            param = network.state_dict()[param_tensor]
            caffeLayerPara = nameDict[param_tensor]
            #print(param_tensor, caffeLayerPara)
            if "," in caffeLayerPara:
                caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
                caffeLayerMatNum = int(caffeLayerMatNum)
                #print (caffeLayerName)
                if caffeLayerName not in caffeParams:
                    print("caffeLayerName is not in caffe 1")
                #print(caffeLayerName, caffeLayerMatNum)
                if "num_batches_tracked" in param_tensor:
                    caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
                else:
                    print(caffeLayerName, caffeLayerMatNum)
                    caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()
            i += 1
        net.save(modelFile) 
        print("net save end")
    if test:
        img = np.ones([1,1,32,256])
        img = img.astype(np.float32)
        timg = np.ones([1,32,256])
        timg = timg.astype(np.float32)
        network.cuda()
        input_var = Variable(torch.FloatTensor(torch.from_numpy(img)).cuda())
        output_var = network(input_var)
        torch.onnx.export(network, input_var, "next.onnx")
        model = network.cnn[0]#.0_layer.layers[0]
        #model2 = model.0_layer
        with torch.no_grad():
            out2=model(input_var)
            out3 = network(input_var)
        print (network)
        torch.onnx.export(model, input_var, "model.onnx")
        #timg = np.transpose(timg, (2, 0, 1)) 
        net.blobs['data'].data[...] = timg
        outc=net.forward()#['cnn_layer']
        caffeout=net.blobs['prob'].data
        out2=out2.cpu().data.numpy()
        print(out2.shape,caffeout.shape)
        print(type(caffeout))
        out2=out2.reshape(-1)
        caffeout=caffeout.reshape(-1)
        print(out2.shape,caffeout.shape)
        for i in range(out2.shape[0]):
            if(abs(out2[i]-caffeout[i])>0.01):
                print("error")
        print(out2)
        print(caffeout)
        sys.exit()

