import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#import models.layer as layer
from models.layer.blocks import RegularBottleneck, DownsamplingBottleneck,DownsamplingBottleneck2, InitialBlock, InputProjectionA, \
    DilatedParallelResidualBlockB, DownSamplerB, C, CBR, BR
from models.layer.blocks import UpsamplingBottleneck,UpsamplingBottleneck2, UpsamplingBottleneck3, RFBblock2

class LANENet(nn.Module):
    def __init__(self, num_classes=2):
        super(LANENet, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.initial_block = InitialBlock(3, 16, padding=1, relu=False)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

        '''
        # Stage 3_1 - Encoder
        self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        '''

        # Stage 4_1 - Decoder
        self.upsample4_0_1 = UpsamplingBottleneck(
            64, 32,  padding=1, dropout_prob=0.1, relu=False)
        self.regular4_1_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)
        self.regular4_2_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)

        #deconv2_down
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )


        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.fc_input_feature = num_classes * int(128 / 8) * int(64 / 8)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5,2,1,1,1]))
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2,2,2,2,2]))
        #self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None):
        x = self.initial_block(img)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        '''
        # Stage 3_1 - Encoder
        x = self.regular3_0_1(x)
        x = self.dilated3_1_1(x)
        x = self.asymmetric3_2_1(x)
        x = self.dilated3_3_1(x)
        x = self.regular3_4_1(x)
        x = self.dilated3_5_1(x)
        x = self.asymmetric3_6_1(x)
        x = self.dilated3_7_1(x)
        #x = self.backbone(img)
        '''

        # Stage 4_1 - Decoder
        x = self.upsample4_0_1(x, max_indices2_0)
        x = self.regular4_1_1(x)
        x = self.regular4_2_1(x)
        #self.feature = x

        out = self.conv_out(x)
        deconv = self.deconv2(out)
        seg_pred = self.deconv3(deconv)

        if seg_gt is not None:
            loss = self.ce_loss(seg_pred, seg_gt)
        else:
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        
        return seg_pred, loss



# class LANENet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(LANENet, self).__init__()

#         self.scale_background = 0.4
#         self.scale_seg = 1.0
#         self.scale_exist = 0.1

#         self.initial_block = InitialBlock(3, 16, padding=1, relu=False)

#         # Stage 1 - Encoder
#         self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
#                                                     relu=False)
#         self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
#         self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
#         self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
#         self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)

#         # Stage 2 - Encoder
#         self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
#                                                     relu=False)
#         self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
#         self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
#         self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
#                                                relu=False)
#         self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
#         self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
#         self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
#         self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
#                                                relu=False)
#         self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

#         '''
#         # Stage 3_1 - Encoder
#         self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
#         self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
#         self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
#                                                relu=False)
#         self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
#         self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
#         self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
#         self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
#                                                relu=False)
#         self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
#         '''

#         # Stage 4_1 - Decoder
#         self.upsample4_0_1 = UpsamplingBottleneck(
#             64, 32,  padding=1, dropout_prob=0.1, relu=False)
#         self.regular4_1_1 = RegularBottleneck(
#             32,  padding=1, dropout_prob=0.1, relu=False)
#         self.regular4_2_1 = RegularBottleneck(
#             32,  padding=1, dropout_prob=0.1, relu=False)

#         #deconv2_down
#         self.conv_out = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#         )

#         #deconv2
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU()  # (nB, 128, 36, 100)
#         )


#         self.deconv3 = nn.Sequential(
#             nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
#         )

#         self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
#         self.fc_input_feature = num_classes * int(128 / 8) * int(64 / 8)
#         #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2, 2]))
#         self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
#         #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2,2,2,2,2]))
#         #self.bce_loss = nn.BCELoss()

#     def forward(self, img, seg_gt=None):
#         x = self.initial_block(img)

#         # Stage 1 - Encoder
#         x, max_indices1_0 = self.downsample1_0(x)
#         x = self.regular1_1(x)
#         x = self.regular1_2(x)
#         x = self.regular1_3(x)
#         x = self.regular1_4(x)

#         # Stage 2 - Encoder
#         x, max_indices2_0 = self.downsample2_0(x)
#         x = self.regular2_1(x)
#         x = self.dilated2_2(x)
#         x = self.asymmetric2_3(x)
#         x = self.dilated2_4(x)
#         x = self.regular2_5(x)
#         x = self.dilated2_6(x)
#         x = self.asymmetric2_7(x)
#         x = self.dilated2_8(x)

#         '''
#         # Stage 3_1 - Encoder
#         x = self.regular3_0_1(x)
#         x = self.dilated3_1_1(x)
#         x = self.asymmetric3_2_1(x)
#         x = self.dilated3_3_1(x)
#         x = self.regular3_4_1(x)
#         x = self.dilated3_5_1(x)
#         x = self.asymmetric3_6_1(x)
#         x = self.dilated3_7_1(x)
#         #x = self.backbone(img)
#         '''

#         # Stage 4_1 - Decoder
#         x = self.upsample4_0_1(x, max_indices2_0)
#         x = self.regular4_1_1(x)
#         x = self.regular4_2_1(x)
#         #self.feature = x

#         out = self.conv_out(x)
#         deconv = self.deconv2(out)
#         seg_pred = self.deconv3(deconv)

#         if seg_gt is not None:
#             loss = self.ce_loss(seg_pred, seg_gt)
#         else:
#             loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        
#         return seg_pred, loss



class BeltNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BeltNet, self).__init__()

        self.initial_block = InitialBlock(3, 16, padding=1, relu=False)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

        '''
        # Stage 3_1 - Encoder
        self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        '''


        # #deconv2_down
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )
        self.regular3_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular3_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        #deconv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )
        self.regular4_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular4_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )
        self.regular5_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular5_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )



    def forward(self, img):
        x = self.initial_block(img)
        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        x = self.conv1(x)
        x = self.regular3_1(x)
        x = self.regular3_2(x)
        x = self.conv2(x)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.conv3(x)
        x = self.regular5_1(x)
        x = self.regular5_2(x)
        x = self.conv4(x)
        #print(x.shape)
        return x




class LANENOVANet(nn.Module):
    def __init__(self, input_size, num_classes=2, pretrained=False):
        super(LANENOVANet, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

        '''
        # Stage 3_1 - Encoder
        self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        '''

        # Stage 4_1 - Decoder
        # self.upsample4_0_1 = UpsamplingBottleneck2(
        #     64, 32,  padding=1, dropout_prob=0.1, relu=False)
        self.upsample4_0_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU() ) # (nB, 128, 36, 100)
        self.regular4_1_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)
        self.regular4_2_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)

        #deconv2_down
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )


        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.fc_input_feature = num_classes * int(128 / 8) * int(64 / 8)
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 2,1]))
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2,2,2,2,2]))
        #self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None):
        x = self.initial_block(img)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        '''
        # Stage 3_1 - Encoder
        x = self.regular3_0_1(x)
        x = self.dilated3_1_1(x)
        x = self.asymmetric3_2_1(x)
        x = self.dilated3_3_1(x)
        x = self.regular3_4_1(x)
        x = self.dilated3_5_1(x)
        x = self.asymmetric3_6_1(x)
        x = self.dilated3_7_1(x)
        #x = self.backbone(img)
        '''

        # Stage 4_1 - Decoder
        x = self.upsample4_0_1(x)
        x = self.regular4_1_1(x)
        x = self.regular4_2_1(x)
        #self.feature = x

        out = self.conv_out(x)
        deconv = self.deconv2(out)
        seg_pred = self.deconv3(deconv)

        if seg_gt is not None:
            loss = self.ce_loss(seg_pred, seg_gt)
        else:
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        
        return seg_pred, loss


class LANENOVANet2(nn.Module):
    def __init__(self, input_size, num_classes=2, pretrained=False):
        super(LANENOVANet2, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

        '''
        # Stage 3_1 - Encoder
        self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        '''

        # Stage 4_1 - Decoder
        self.upsample4_0_1 = UpsamplingBottleneck2(
            64, 32,  padding=1, dropout_prob=0.1, relu=False)
        # self.upsample4_0_1 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU() ) # (nB, 128, 36, 100)
        self.regular4_1_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)
        self.regular4_2_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)

        #deconv2_down
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )


        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.fc_input_feature = num_classes * int(128 / 8) * int(64 / 8)
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 2,1]))
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2,2,2,2,2]))
        #self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None):
        x = self.initial_block(img)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        '''
        # Stage 3_1 - Encoder
        x = self.regular3_0_1(x)
        x = self.dilated3_1_1(x)
        x = self.asymmetric3_2_1(x)
        x = self.dilated3_3_1(x)
        x = self.regular3_4_1(x)
        x = self.dilated3_5_1(x)
        x = self.asymmetric3_6_1(x)
        x = self.dilated3_7_1(x)
        #x = self.backbone(img)
        '''

        # Stage 4_1 - Decoder
        x = self.upsample4_0_1(x)
        x = self.regular4_1_1(x)
        x = self.regular4_2_1(x)
        #self.feature = x

        out = self.conv_out(x)
        deconv = self.deconv2(out)
        seg_pred = self.deconv3(deconv)

        if seg_gt is not None:
            loss = self.ce_loss(seg_pred, seg_gt)
        else:
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        
        return seg_pred, loss



class LANENOVANet3(nn.Module):
    def __init__(self, input_size, num_classes=2, pretrained=False):
        super(LANENOVANet3, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.r1 = RFBblock2(16, 2)
        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=False)
        self.r2 = RFBblock2(32, 2)
        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=False)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_2 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric2_3 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_4 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated2_6 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric2_7 = RegularBottleneck(64, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)

        '''
        # Stage 3_1 - Encoder
        self.regular3_0_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_1_1 = RegularBottleneck(64,  dilation=2, padding=2, dropout_prob=0.1, relu=False)
        self.asymmetric3_2_1 = RegularBottleneck(64,  kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_3_1 = RegularBottleneck(64,  dilation=4, padding=4, dropout_prob=0.1, relu=False)
        self.regular3_4_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        self.dilated3_5_1 = RegularBottleneck(64,  dilation=8, padding=8, dropout_prob=0.1, relu=False)
        self.asymmetric3_6_1 = RegularBottleneck(64,  kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=False)
        self.dilated3_7_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=False)
        '''

        # Stage 4_1 - Decoder
        self.upsample4_0_1 = UpsamplingBottleneck2(
            64, 32,  padding=1, dropout_prob=0.1, relu=False)
        # self.upsample4_0_1 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU() ) # (nB, 128, 36, 100)
        self.regular4_1_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)
        self.regular4_2_1 = RegularBottleneck(
            32,  padding=1, dropout_prob=0.1, relu=False)

        #deconv2_down
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()  # (nB, 128, 36, 100)
        )


        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.fc_input_feature = num_classes * int(128 / 8) * int(64 / 8)
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 2,1]))
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2,2,2,2,2]))
        #self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None):
        x = self.initial_block(img)
        x = self.r1(x)
        
        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x = self.r2(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        '''
        # Stage 3_1 - Encoder
        x = self.regular3_0_1(x)
        x = self.dilated3_1_1(x)
        x = self.asymmetric3_2_1(x)
        x = self.dilated3_3_1(x)
        x = self.regular3_4_1(x)
        x = self.dilated3_5_1(x)
        x = self.asymmetric3_6_1(x)
        x = self.dilated3_7_1(x)
        #x = self.backbone(img)
        '''

        # Stage 4_1 - Decoder
        x = self.upsample4_0_1(x)

        x = self.regular4_1_1(x)
        x = self.regular4_2_1(x)
        #self.feature = x

        out = self.conv_out(x)
        deconv = self.deconv2(out)
        seg_pred = self.deconv3(deconv)
        if seg_gt is not None:
            loss = self.ce_loss(seg_pred, seg_gt)
        else:
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)        
        return seg_pred, loss