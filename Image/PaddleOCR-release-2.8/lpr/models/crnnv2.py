import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
# sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from models.base.bsltm import BidirectionalLSTM, BidirectionalLSTM2
import models.layer as layer
class CRNN(nn.Module):

    def __init__(self,  nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 2)        #128 32 160 16
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #128 32 80 8
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #128 32 80 8
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #64 16 40 4
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #64 16 40 4
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #32 8 20 2
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #16 4 10 1
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2

        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.rnn = nn.Sequential(
            BidirectionalLSTM(out_channel, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.cnn_layer = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd

    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        assert h == 1, "the height of conv must be 1"
        conv=self.cnn_layer(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(conv, dim=2)
        return output


class CRNNv1(nn.Module):

    def __init__(self,  nclass, leakyRelu=False):
        super(CRNNv1, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 2)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #128 32 80 8
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #128 32 80 8
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #64 16 40 4
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #64 16 40 4
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #32 8 20 2
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #16 4
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2
        layer_list = [
            OrderedDict([ 
                ('0_0_layer',      layer0_0),        #0
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.rnn = nn.Sequential(
            BidirectionalLSTM(out_channel, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.cnn_layer = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd

    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        b, c, h, w = conv.size()
        conv = conv.view(b,c, 1, h*w)
        #conv=self.cnn_layer(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return conv


class CRNNv3(nn.Module):
    def __init__(self,  nclass, leakyRelu=False):
        super(CRNNv3, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 2)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #64 16
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #64 16
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #32 8
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #32 8
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #16 4
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #8 2 
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2
        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.rnn = nn.Sequential(
            BidirectionalLSTM(out_channel, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.cnn_layer = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd

    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        #assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, c, 1, h*w)
        conv = self.cnn_layer(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        #conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output


class CRNNv3_BGR(nn.Module):
    def __init__(self,  nclass, leakyRelu=False):
        super(CRNNv3_BGR, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(3, 16, 3, 2)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #64 16
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #64 16
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #32 8
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #32 8
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #16 4
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #8 2 
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2
        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.rnn = nn.Sequential(
            BidirectionalLSTM(out_channel, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.cnn_layer = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd

    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        #assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, c, 1, h*w)
        conv = self.cnn_layer(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        #conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output


class CRNNv4(nn.Module):
    def __init__(self,  nclass, leakyRelu=False):
        super(CRNNv4, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 1)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #64 16
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #64 16
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #32 8
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #32 8
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #16 4
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #8 2 
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2
        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.rnn = nn.Sequential(
            nn.Conv2d(256, 256, (2,1), 1, padding=0),
            nn.BatchNorm2d(256), 
            nn.ReLU())
        self.cnn_layer_new = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd
    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        conv =  self.rnn(conv)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        #assert h == 1, "the height of conv must be 1"
        conv = conv.view(b,c, 1, h*w)
        conv=self.cnn_layer_new(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        #conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output

class CRNNv5(nn.Module):
    def __init__(self,  nclass, leakyRelu=False):
        super(CRNNv5, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        #cfg.append([1, 512, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 1)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #64 16
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #64 16
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #32 8
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #32 8
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #16 4
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #8 2 
        #layer6, out_channel= layer.buildInvertedResBlock(cfg[5], 256)    #8 2
        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
                #('6_layer',      layer6),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, (2,2), stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.rnn = nn.Sequential(
            BidirectionalLSTM2(256,256,nclass))
        self.cnn_layer_new = nn.Conv2d(256, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd
    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        conv=self.cnn2(conv)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        #assert h == 1, "the height of conv must be 1"
        #conv = conv.view(b,c, 1, h*w)
        #conv=self.cnn_layer_new(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output


class TCBlock(nn.Module):
    expansion = 1
    conv_kernel = (1, 3)
    conv_padding = (0, 1)

    def __init__(self, in_planes, planes, stride=1):
        super(TCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=self.conv_kernel,
                               stride=stride, padding=self.conv_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.conv_kernel,
                               stride=1, padding=self.conv_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.relu = torch.nn.ReLU()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class CRNNv6(nn.Module):
    def __init__(self, nclass):
        super(CRNNv6, self).__init__()
        nh=128
        cfg = []
        cfg.append([1, 32, 1, 1]) #t, c, n, s
        cfg.append([1, 32, 2, 2])
        cfg.append([2, 64, 2, 1])
        cfg.append([2, 128, 2, 2])
        cfg.append([1, 256, 2, 2])
        layer0_0 = layer.Conv2dBatchReLU(1, 16, 3, 2)        #128 32
        layer0 = layer.Conv2dBatchReLU(16, 32, 3, 2)        #64 16
        layer1, _= layer.buildInvertedResBlock(cfg[0], 32)    #64 16
        layer2, _= layer.buildInvertedResBlock(cfg[1], 32)    #32 8
        layer3, _= layer.buildInvertedResBlock(cfg[2], 32)    #32 8
        layer4, _= layer.buildInvertedResBlock(cfg[3], 64)    #16 4
        layer5, out_channel= layer.buildInvertedResBlock(cfg[4], 128)    #8 2 
        layer_list = [
            OrderedDict([                         #0
                ('0_0_layer',      layer0_0),
                ('0_layer',      layer0),
                ('1_layer',      layer1),
                ('2_layer',      layer2),
                ('3_layer',      layer3),
                ('4_layer',      layer4),
                ('5_layer',      layer5),
            ]),
        ]
        self.cnn = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        self.planes = [128, 128]
        self.in_planes = 256
        self.layer1_1 = self._make_layer(TCBlock, int(self.planes[0]), stride=1)
        self.layer2_1 = self._make_layer(TCBlock, int(self.planes[1]), stride=1)
        self.layer2_2 = self._make_layer(TCBlock, int(self.planes[1]), stride=1)

        self.cnn_layer = nn.Conv2d(128, nclass, 1, 1, 0)#cnn_layer#cnn_layer_zd

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        # conv features
        conv = self.cnn[0](input)
        b, c, h, w = conv.size()
        conv = conv.view(b, c, 1, h*w)

        out = self.layer1_1(conv)
        out = self.layer2_1(out)
        conv = self.layer2_2(out)

        conv = self.cnn_layer(conv)

        # return conv
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output
    

### function layers
class RFBblock(nn.Module):
    def __init__(self, C, grp=2):
        super(RFBblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=1, dilation=1, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=5, dilation=5, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=7, dilation=7, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )
    def forward(self, x):
        x1 = self.conv1(x) # 3x3 conv + 3x3 conv rate=1
        x3 = self.conv3(x) # 3x3 conv + 3x3 conv rate=3
        x5 = self.conv5(x) # 3x3 conv + 3x3 conv rate=5
        x7 = self.conv7(x) # 3x3 conv + 3x3 conv rate=7
        xc = torch.cat((x1, x3, x5, x7), 1)
        x  = self.conv0(x + xc)
        return x

class RFBblock2(nn.Module):
    def __init__(self, C, grp=2):
        super(RFBblock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=1, dilation=1, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=5, dilation=5, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(C, C//4, kernel_size=3, stride=1, padding=7, dilation=7, groups=grp),
            nn.BatchNorm2d(C//4),
            nn.ReLU(),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )
    def forward(self, x):
        x1 = self.conv1(x) # 3x3 conv + 3x3 conv rate=1
        x3 = self.conv3(x) # 3x3 conv + 3x3 conv rate=3
        x5 = self.conv5(x) # 3x3 conv + 3x3 conv rate=5
        x7 = self.conv7(x) # 3x3 conv + 3x3 conv rate=7
        xc = torch.cat((x1, x3, x5, x7), 1)
        x  = self.conv0(x + xc)
        return x

### backbone
class baselayers(nn.Module):
    def __init__(self, fc, n_class, grp=4):
        super(baselayers, self).__init__()
        self.conv0 = nn.Sequential(
            ### first conv block
            nn.Conv2d(3, fc, 3, 1, 1),
            nn.BatchNorm2d(fc),
            nn.ReLU(),
            nn.Conv2d(fc, fc, 3, 1, 1),
            nn.BatchNorm2d(fc),
            nn.ReLU(),

            nn.Conv2d(fc, fc*2, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*2), 
            nn.ReLU(), 
            nn.Conv2d(fc*2, fc*2, 3, 1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*2), 
            nn.ReLU(), 
            ### downsample block1: -1, 2fc, 1/2, 1/2

            nn.Conv2d(fc*2, fc*4, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*4), 
            nn.ReLU(), 
            nn.Conv2d(fc*4, fc*4, 3, 1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*4), 
            nn.ReLU(), 
            ### downsample block2: -1, 4fc, 1/4, 1/4

            nn.Conv2d(fc*4, fc*8, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*8), 
            nn.ReLU(), 
            nn.Conv2d(fc*8, fc*8, 3, 1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*8), 
            nn.ReLU(), 
            ### downsample block3: -1, 8fc, 1/8, 1/8
            
            nn.Conv2d(fc*8, fc*12, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(),
            nn.Conv2d(fc*12, fc*12, 3, 1, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(), 
            ### downsample block4: -1, 12fc, 1/16, 1/16

            nn.Conv2d(fc*12, fc*12, (2,1), 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(), 
            nn.Conv2d(fc*12, fc*12, (3,1), 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12),
            nn.ReLU(), 
            ### downsample block5: -1, 12fc, 1/32, 1/16

            nn.Conv2d(fc*12, n_class, 1, 1, padding=0),
            ### output layer: -1, n_class, 1/32, 1/16
        )
    def forward(self, x):
        c0 = self.conv0(x)
        return c0

class rfblayers(nn.Module):
    def __init__(self, fc, n_class, grp=4):
        super(rfblayers, self).__init__()
        self.d0 = nn.Sequential(
            ### first conv block
            nn.Conv2d(3, fc, 3, 1, 1),
            nn.BatchNorm2d(fc),
            nn.ReLU(),
            nn.Conv2d(fc, fc, 3, 1, 1),
            nn.BatchNorm2d(fc),
            nn.ReLU(),
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(fc, fc*2, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*2), 
            nn.ReLU(),
            ### downsample block1: -1, 2fc, 1/2, 1/2
        )
        self.r1 = RFBblock(fc*2, 2)

        self.d2 = nn.Sequential(
            nn.Conv2d(fc*2, fc*4, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*4), 
            nn.ReLU(),
            ### downsample block2: -1, 4fc, 1/4, 1/4
        )
        self.r2 = RFBblock(fc*4, 2)

        self.d3 = nn.Sequential(
            nn.Conv2d(fc*4, fc*8, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*8), 
            nn.ReLU(),
            ### downsample block3: -1, 8fc, 1/8, 1/8
        )
        self.r3 = RFBblock(fc*8, 4)

        self.d4 = nn.Sequential(
            nn.Conv2d(fc*8, fc*12, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(),
            ### downsample block4: -1, 12fc, 1/16, 1/16
        )
        self.r4 = RFBblock(fc*12, 4)

        self.head = nn.Sequential(
            nn.Conv2d(fc*12, fc*12, (2,1), 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(), 
            nn.Conv2d(fc*12, fc*12, (3,1), 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12),
            nn.ReLU(), 
            ### head block: -1, 12fc, 1/32, 1/16
            nn.Conv2d(fc*12, n_class, 1, 1, padding=0)
            ### output layer: -1, n_class, 1/32, 1/16
        )

    def forward(self, x):
        c = self.d0(x)
        c = self.r1(self.d1(c))
        c = self.r2(self.d2(c))
        c = self.r3(self.d3(c))
        c = self.r4(self.d4(c))
        o = self.head(c)
        return o

class rfbslimc(nn.Module):
    # rfbblock1
    # 4x3 head
    def __init__(self, fc, n_class, grp=4):
        super(rfbslimc, self).__init__()
        self.d0 = nn.Sequential(
            ### first conv block
            nn.Conv2d(1, fc, 3, 1, 1),
            nn.BatchNorm2d(fc),
            nn.ReLU(),
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(fc, fc*2, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*2), 
            nn.ReLU(),
            ### downsample block1: -1, 2fc, 1/2, 1/2
            RFBblock2(fc*2, 2)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(fc*2, fc*4, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*4), 
            nn.ReLU(),
            ### downsample block2: -1, 4fc, 1/4, 1/4
            RFBblock2(fc*4, 2)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(fc*4, fc*8, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*8), 
            nn.ReLU(),
            ### downsample block3: -1, 8fc, 1/8, 1/8
            RFBblock2(fc*8, 4),
            nn.Dropout(0.4),
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(fc*8, fc*12, 3, 2, padding=3, dilation=3, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(),
            ### downsample block4: -1, 12fc, 1/16, 1/16
            RFBblock2(fc*12, 4),
            nn.Dropout(0.4),
        )

        self.head = nn.Sequential(
            nn.Conv2d(fc*12, fc*12, 3, 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12), 
            nn.ReLU(), 
            nn.Conv2d(fc*12, fc*12, 2, 1, padding=0, groups=grp),
            nn.BatchNorm2d(fc*12),
            nn.ReLU(), 
            ### head block: -1, 12fc, 1/32, 1/16
            nn.Conv2d(fc*12, n_class, 1, 1, padding=0)
            ### output layer: -1, n_class, 1/32, 1/16
        )

    def forward(self, x):
        x = self.d0(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        conv = self.d4(x)
        # #assert h == 1, "the height of conv must be 1"
        # conv = conv.view(b,c, 1, h*w)
        conv=o = self.head(conv)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print (conv.shape)
        # rnn features
        #conv = self.rnn(conv)
        output = F.log_softmax(conv, dim=2)
        #print (output.shape)
        return output
# import torch
# import numpy as np
# from torch.autograd import Variable
# from torchsummary import summary
# #model = CRNN(32,3,69,128)
# model = CRNN2(69)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = model.to(device)
# img = np.ones([1,3,64,256])
# img = img.astype(np.float32)
# input_var = Variable(torch.FloatTensor(torch.from_numpy(img)).cuda())
# model.eval()
# _=model(input_var)
# #summary(model, (3, 32, 128))
