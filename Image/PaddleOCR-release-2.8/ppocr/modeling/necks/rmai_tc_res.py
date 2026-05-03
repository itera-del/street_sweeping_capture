from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr, reshape, transpose
from paddle import nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay
import paddle.nn.functional as F


class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class TCBlock(nn.Layer):
    expansion = 1
    conv_kernel = (1, 3)
    conv_padding = (0, 1)

    def __init__(self, in_planes, planes, stride=1):
        super(TCBlock, self).__init__()

        # normal
        self.conv1 = Conv2D(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=self.conv_kernel,
            stride=stride,
            padding=self.conv_padding,
            groups=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn1 = BatchNorm(
            planes,
            act='relu',
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.conv2 = Conv2D(
            in_channels=planes,
            out_channels=planes,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=self.conv_padding,
            groups=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn2 = BatchNorm(
            planes,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(

                # normal
                Conv2D(
                    in_channels=in_planes,
                    out_channels=self.expansion*planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=1,
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                BatchNorm(
                    self.expansion*planes,
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            )

    def forward(self, x):
        # normal
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        out = paddle.add(y, self.shortcut(x))
        out = F.relu(out)
        return out


class TCBlockMobile(nn.Layer):
    expansion = 1
    conv_kernel = (1, 3)
    conv_padding = (0, 1)

    def __init__(self, in_planes, planes, stride=1):
        super(TCBlockMobile, self).__init__()

        # mobile net
        if abs(in_planes - planes) < .01:

            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(
                    in_channels=in_planes, 
                    out_channels=planes, 
                    kernel_size=self.conv_kernel,
                    stride=stride, 
                    padding=self.conv_padding,
                    groups=planes, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm(
                    num_channels=planes, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # pw-linear
                nn.Conv2D(
                    in_channels=planes, 
                    out_channels=planes, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm(
                    num_channels=planes, 
                    act=None,
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2D(
                    in_channels=in_planes, 
                    out_channels=planes, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm(
                    num_channels=planes, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # dw
                nn.Conv2D(
                    in_channels=planes, 
                    out_channels=planes, 
                    kernel_size=self.conv_kernel,
                    stride=stride, 
                    padding=self.conv_padding,
                    groups=planes, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm(
                    num_channels=planes, 
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                # pw-linear
                nn.Conv2D(
                    in_channels=planes, 
                    out_channels=planes, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    groups=1, 
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm(
                    num_channels=planes, 
                    act=None,
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            )

        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(

                # normal
                Conv2D(
                    in_channels=in_planes,
                    out_channels=self.expansion*planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=1,
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                BatchNorm(
                    self.expansion*planes,
                    act='relu',
                    param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            )

    def forward(self, x):
        # mobile net
        y = self.conv(x)

        out = paddle.add(y, self.shortcut(x))
        out = F.relu(out)
        return out


class EncoderWithTCRES(nn.Layer):
    def __init__(self, in_channels, hidden_size, widen_factor=1.0, deepen_factor=1.0):
        super(EncoderWithTCRES, self).__init__()

        self.out_channels = hidden_size

        self.planes = [in_channels, hidden_size]
        self.in_planes = self.planes[0]
        self.layer1_1 = self._make_layer(TCBlock, int(self.planes[0] * widen_factor), stride=1)
        self.layer1_2 = self._make_layer(TCBlock, int(self.planes[0] * widen_factor), stride=1)
        self.layer2_1 = self._make_layer(TCBlock, int(self.planes[1] * widen_factor), stride=1)
        self.layer2_2 = self._make_layer(TCBlock, int(self.planes[1] * widen_factor), stride=1)

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1_1(x)
        out = self.layer1_2(out)
        out = self.layer2_1(out)
        out = self.layer2_2(out)

        return out


class EncoderWithTCRES_Mobile(nn.Layer):
    def __init__(self, in_channels, hidden_size, widen_factor=1.0, deepen_factor=1.0):
        super(EncoderWithTCRES_Mobile, self).__init__()

        self.out_channels = int(hidden_size * widen_factor)
        self.deepen_factor = deepen_factor

        self.planes = [in_channels, hidden_size]
        self.in_planes = self.planes[0]
        self.layer1_1 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
        if self.deepen_factor == 2.0:
            self.layer1_2 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
        elif self.deepen_factor == 3.0:
            self.layer1_2 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
            self.layer1_3 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
        elif self.deepen_factor == 4.0:
            self.layer1_2 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
            self.layer1_3 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
            self.layer1_4 = self._make_layer(TCBlockMobile, int(self.planes[0] * widen_factor), stride=1)
        self.layer2_1 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
        self.layer2_2 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
        if self.deepen_factor == 1.5:
            self.layer2_3 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
        elif self.deepen_factor == 2.0:
            self.layer2_3 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_4 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
        elif self.deepen_factor == 3.0:
            self.layer2_3 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_4 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_5 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_6 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
        elif self.deepen_factor == 4.0:
            self.layer2_3 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_4 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_5 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_6 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_7 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)
            self.layer2_8 = self._make_layer(TCBlockMobile, int(self.planes[1] * widen_factor), stride=1)

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("inputs shape: ", x.shape)
        out = self.layer1_1(x)
        # print("layer1_1 shape: ", out.shape)
        if self.deepen_factor == 2.0:
            out = self.layer1_2(out)
        elif self.deepen_factor == 3.0:
            out = self.layer1_2(out)
            out = self.layer1_3(out)
        elif self.deepen_factor == 4.0:
            out = self.layer1_2(out)
            out = self.layer1_3(out)
            out = self.layer1_4(out)
        out = self.layer2_1(out)
        # print("layer2_1 shape: ", out.shape)
        out = self.layer2_2(out)
        # print("layer2_2 shape: ", out.shape)
        if self.deepen_factor == 1.5:
            out = self.layer2_3(out)
        elif self.deepen_factor == 2.0:
            out = self.layer2_3(out)
            out = self.layer2_4(out)
        elif self.deepen_factor == 3.0:
            out = self.layer2_3(out)
            out = self.layer2_4(out)
            out = self.layer2_5(out)
            out = self.layer2_6(out)
        elif self.deepen_factor == 4.0:
            out = self.layer2_3(out)
            out = self.layer2_4(out)
            out = self.layer2_5(out)
            out = self.layer2_6(out)
            out = self.layer2_7(out)
            out = self.layer2_8(out)
        # print("EncoderWithTCRES_Mobile output shape: ", out.shape)

        return out


class TCRES(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, widen_factor=1.0, deepen_factor=1.0, **kwargs):
        super(TCRES, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type

        support_encoder_dict = {
            'tcres': EncoderWithTCRES,
            'tcres_mobile': EncoderWithTCRES_Mobile,
            'tcres_mobile_no_reshape': EncoderWithTCRES_Mobile,
        }
        assert encoder_type in support_encoder_dict, '{} must in {}'.format(
            encoder_type, support_encoder_dict.keys())

        self.encoder = support_encoder_dict[encoder_type](
            self.encoder_reshape.out_channels, hidden_size, widen_factor, deepen_factor)

        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        if self.encoder_type == 'tcres_mobile_no_reshape':
            x = self.encoder(x)
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
        return x
