import math
import numpy as np
import paddle
from paddle import ParamAttr, reshape, transpose
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay
from paddle.nn.functional import hardswish, hardsigmoid


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu'):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, isBias=False):
        super(InvertedResidual,self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup
        self.isBias = isBias
        if abs(expand_ratio - 1) < .01:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(
                    in_channels=inp, 
                    out_channels=inp * expand_ratio, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=1, 
                    groups=inp * expand_ratio, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=self.isBias),
                nn.BatchNorm(
                    num_channels=inp * expand_ratio, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # pw-linear
                nn.Conv2D(
                    in_channels=inp * expand_ratio, 
                    out_channels=oup, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=self.isBias),
                nn.BatchNorm(
                    num_channels=oup, 
                    act=None,
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2D(
                    in_channels=inp, 
                    out_channels=inp * expand_ratio, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=self.isBias),
                nn.BatchNorm(
                    num_channels=inp * expand_ratio, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # dw
                nn.Conv2D(
                    in_channels=inp * expand_ratio, 
                    out_channels=inp * expand_ratio, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=1, 
                    groups=inp * expand_ratio, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=self.isBias),
                nn.BatchNorm(
                    num_channels=inp * expand_ratio, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # pw-linear
                nn.Conv2D(
                    in_channels=inp * expand_ratio, 
                    out_channels=oup, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=self.isBias),
                nn.BatchNorm(
                    num_channels=oup, 
                    act=None,
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            )

    def forward(self, x):
        if self.use_res_connect:
            out =  paddle.add(self.conv(x), x)
        else:
            out = self.conv(x)
        return out 


def buildInvertedResBlock(residual_setting, input_channel):
    # building inverted residual blocks
    features = []
    t, c, n, s = residual_setting
    output_channel = int(c)

    for i in range(n):
        if i == 0:
            curLayer = InvertedResidual(input_channel, output_channel, s, t)
            features.append(curLayer)
        else:
            curLayer = InvertedResidual(input_channel, output_channel, 1, t)
            features.append(curLayer)
        input_channel = output_channel
    layers = nn.Sequential(*features)
    return layers, output_channel


class MobileNetV1RM(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 widen_factor=1.0,
                 deepen_factor=1.0,
                 **kwargs):
        super().__init__()

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            filter_size=3,
            num_filters=int(16*widen_factor),
            stride=2,
            padding=1)

        self.conv2 = ConvBNLayer(
            num_channels=int(16*widen_factor),
            filter_size=3,
            num_filters=int(32*widen_factor),
            stride=2,
            padding=1)

        self.conv3, _= buildInvertedResBlock([1, int(32*widen_factor), int(1*deepen_factor), 1], int(32*widen_factor))
        self.conv4, _= buildInvertedResBlock([1, int(32*widen_factor), int(2*deepen_factor), 2], int(32*widen_factor))
        self.conv5, _= buildInvertedResBlock([2, int(64*widen_factor), int(2*deepen_factor), 1], int(32*widen_factor))
        self.conv6, _= buildInvertedResBlock([2, int(128*widen_factor), int(2*deepen_factor), 2], int(64*widen_factor))
        self.conv7, out_channel= buildInvertedResBlock([1, int(256*widen_factor), int(2*deepen_factor), 2], int(128*widen_factor))

        self.out_channels = int(out_channel)

    def forward(self, inputs):
        # print("inputs shape: ", inputs.shape)
        y = self.conv1(inputs)
        # print("conv1 shape: ", y.shape)
        y = self.conv2(y)
        # print("conv2 shape: ", y.shape)
        y = self.conv3(y)
        # print("conv3 shape: ", y.shape)
        y = self.conv4(y)
        # print("shape shape: ", y.shape)
        y = self.conv5(y)
        # print("conv5 shape: ", y.shape)
        y = self.conv6(y)
        # print("conv6 shape: ", y.shape)
        y = self.conv7(y)
        # print("conv7 shape: ", y.shape)

        # # train
        b, c, h, w = y.shape
        y = paddle.reshape(y, [b, c, 1, h*w])
        # print("MobileNetV1RM output shape: ", y.shape)

        # test
        # y = paddle.reshape(y, [1, 256, 1, 32])    # 128 * 256
        # y = paddle.reshape(y, [1, 256, 1, 16])    # 64 * 256
        # y = paddle.reshape(y, [1, 384, 1, 16])    # 64 * 256 widen_factor: 1.5
        # y = paddle.reshape(y, [1, 460, 1, 16])    # 64 * 256 widen_factor: 1.8
        # y = paddle.reshape(y, [1, 256, 1, 20])    # 64 * 320
        return y