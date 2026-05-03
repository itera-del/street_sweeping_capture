
import sys
import torch
import numpy as np
import os
import models.LaneNet as LaneNet
from torch.autograd import Variable
caffe_root = '/home/workspace/pfji/enet_prun/caffe-enet/'
sys.path.insert(0, caffe_root + 'python')  
import caffe


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def save_conv(torch_name, caffe_name, weights, net):
    # if caffe_name == 'conv0_1' :
    #     conv_key = '{}.weight'.format(torch_name)
    #     bias_key = '{}.bias'.format(torch_name)
    #     conv_weight = weights[conv_key].cpu().numpy()
    #     conv_bias = weights[bias_key].cpu().numpy()
    #     net.params[caffe_name][0].data[...] = conv_weight
    #     net.params[caffe_name][1].data[...] = conv_bias
    # else:
    conv_key = '{}.weight'.format(torch_name)
    #print(conv_key)
    conv_weight = weights[conv_key].cpu().numpy()
    net.params[caffe_name][0].data[...] = conv_weight
    #print(conv_weight)

    print("add weight of {} success!".format(caffe_name))


def save_deconv(torch_name, caffe_name, weights, net):
    if caffe_name == 'decon4_0_1':
        #print()
        deconv_key = '{}.weight'.format(torch_name)
        deconv_weight = weights[deconv_key].cpu().numpy()
        net.params[caffe_name][0].data[...] = deconv_weight
    else:
        deconv_key = '{}.weight'.format(torch_name)
        bias_key = '{}.bias'.format(torch_name)
        deconv_weight = weights[deconv_key].cpu().numpy()
        deconv_bias = weights[bias_key].cpu().numpy()
        net.params[caffe_name][0].data[...] = deconv_weight
        net.params[caffe_name][1].data[...] = deconv_bias
        print("add weight of {} success!".format(caffe_name))


def save_batch_norm(torch_name, caffe_name, weights, net):
    mean_key = '{}.running_mean'.format(torch_name)
    var_key = '{}.running_var'.format(torch_name)
    running_mean = weights[mean_key].cpu().numpy()
    running_var = weights[var_key].cpu().numpy()
    net.params[caffe_name][0].data[...] = running_mean
    net.params[caffe_name][1].data[...] = running_var
    net.params[caffe_name][2].data[...] = np.array([1.0])
    print("add weight of {} success!".format(caffe_name))


def save_scale(torch_name, caffe_name, weights, net):
    scale_key = '{}.weight'.format(torch_name)
    bias_key = '{}.bias'.format(torch_name)
    #print(scale_key)
    scale_weight = weights[scale_key].cpu().numpy()
    scale_bias = weights[bias_key].cpu().numpy()
    #print(scale_weight)
    net.params[caffe_name][0].data[...] = scale_weight
    net.params[caffe_name][1].data[...] = scale_bias
    print(scale_bias)
    print("add weight of {} success!".format(caffe_name))


def save_prelu(torch_name, caffe_name, weights, net):
    weight_key = '{}.weight'.format(torch_name)
    prelu_weight = weights[weight_key].cpu().numpy()
    net.params[caffe_name][0].data[...] = prelu_weight
    print("add weight of {} success!".format(caffe_name))

def save_fc(torch_name, caffe_name, weights, net):
    weight_key = '{}.weight'.format(torch_name)
    bias_key = '{}.bias'.format(torch_name)
    fc_weight = weights[weight_key].cpu().numpy()
    fc_bias = weights[bias_key].cpu().numpy()
    net.params[caffe_name][0].data[...] = fc_weight
    net.params[caffe_name][1].data[...] = fc_bias
    print("add weight of {} success!".format(caffe_name))

def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.cpu().data.numpy()
    bn_param[1].data[...] = running_var.cpu().data.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.cpu().data.numpy()
    scale_param[0].data[...] = weights.cpu().data.numpy()

def save_prelu2caffe(weights, prelu_param):
    print(weights.cpu().data.numpy())
    prelu_param[0].data[...] = weights.cpu().data.numpy()



if __name__ == '__main__':
    # build map
    weight_map = [
        ('initial_block.main_branch', 'conv0_1'),
        ('initial_block.batch_norm', 'bn0_1'),
        ('initial_block.batch_norm', 'scale0_1'),
        ('initial_block.out_prelu', 'prelu0_1'),
        ('downsample1_0.ext_conv1.0', 'conv1_0_0'),
        ('downsample1_0.ext_conv1.1', 'bn1_0_0'),
        ('downsample1_0.ext_conv1.1', 'scale1_0_0'),
        ('downsample1_0.ext_conv1.2', 'prelu1_0_0'),
        ('downsample1_0.ext_conv2.0', 'conv1_0_1'),
        ('downsample1_0.ext_conv2.1', 'bn1_0_1'),
        ('downsample1_0.ext_conv2.1', 'scale1_0_1'),
        ('downsample1_0.ext_conv2.2', 'prelu1_0_1'),
        ('downsample1_0.ext_conv3.0', 'conv1_0_2'),
        ('downsample1_0.ext_conv3.1', 'bn1_0_2'),
        ('downsample1_0.ext_conv3.1', 'scale1_0_2'),
        ('downsample1_0.ext_conv4.0', 'conv1_0_4'),
        ('downsample1_0.ext_conv4.1', 'bn1_0_4'),
        ('downsample1_0.ext_conv4.1', 'scale1_0_4'),
        ('downsample1_0.ext_conv4.2', 'prelu1_0_3'),
        ('downsample1_0.out_prelu', 'prelu1_0_4'),
        ('regular1_1.ext_conv1.0', 'conv1_1_0'),
        ('regular1_1.ext_conv1.1', 'bn1_1_0'),
        ('regular1_1.ext_conv1.1', 'scale1_1_0'),
        ('regular1_1.ext_conv1.2', 'prelu1_1_0'),
        ('regular1_1.ext_conv2.0', 'conv1_1_1'),
        ('regular1_1.ext_conv2.1', 'bn1_1_1'),
        ('regular1_1.ext_conv2.1', 'scale1_1_1'),
        ('regular1_1.ext_conv2.2', 'prelu1_1_1'),
        ('regular1_1.ext_conv3.0', 'conv1_1_2'),
        ('regular1_1.ext_conv3.1', 'bn1_1_2'),
        ('regular1_1.ext_conv3.1', 'scale1_1_2'),
        ('regular1_1.out_prelu', 'prelu1_1_4'),
        ('regular1_2.ext_conv1.0', 'conv1_2_0'),
        ('regular1_2.ext_conv1.1', 'bn1_2_0'),
        ('regular1_2.ext_conv1.1', 'scale1_2_0'),
        ('regular1_2.ext_conv1.2', 'prelu1_2_0'),
        ('regular1_2.ext_conv2.0', 'conv1_2_1'),
        ('regular1_2.ext_conv2.1', 'bn1_2_1'),
        ('regular1_2.ext_conv2.1', 'scale1_2_1'),
        ('regular1_2.ext_conv2.2', 'prelu1_2_1'),
        ('regular1_2.ext_conv3.0', 'conv1_2_2'),
        ('regular1_2.ext_conv3.1', 'bn1_2_2'),
        ('regular1_2.ext_conv3.1', 'scale1_2_2'),
        ('regular1_2.out_prelu', 'prelu1_2_4'),
        ('regular1_3.ext_conv1.0', 'conv1_3_0'),
        ('regular1_3.ext_conv1.1', 'bn1_3_0'),
        ('regular1_3.ext_conv1.1', 'scale1_3_0'),
        ('regular1_3.ext_conv1.2', 'prelu1_3_0'),
        ('regular1_3.ext_conv2.0', 'conv1_3_1'),
        ('regular1_3.ext_conv2.1', 'bn1_3_1'),
        ('regular1_3.ext_conv2.1', 'scale1_3_1'),
        ('regular1_3.ext_conv2.2', 'prelu1_3_1'),
        ('regular1_3.ext_conv3.0', 'conv1_3_2'),
        ('regular1_3.ext_conv3.1', 'bn1_3_2'),
        ('regular1_3.ext_conv3.1', 'scale1_3_2'),
        ('regular1_3.out_prelu', 'prelu1_3_4'),
        ('regular1_4.ext_conv1.0', 'conv1_4_0'),
        ('regular1_4.ext_conv1.1', 'bn1_4_0'),
        ('regular1_4.ext_conv1.1', 'scale1_4_0'),
        ('regular1_4.ext_conv1.2', 'prelu1_4_0'),
        ('regular1_4.ext_conv2.0', 'conv1_4_1'),
        ('regular1_4.ext_conv2.1', 'bn1_4_1'),
        ('regular1_4.ext_conv2.1', 'scale1_4_1'),
        ('regular1_4.ext_conv2.2', 'prelu1_4_1'),
        ('regular1_4.ext_conv3.0', 'conv1_4_2'),
        ('regular1_4.ext_conv3.1', 'bn1_4_2'),
        ('regular1_4.ext_conv3.1', 'scale1_4_2'),
        ('regular1_4.out_prelu', 'prelu1_4_4'),
        ('downsample2_0.ext_conv1.0', 'conv2_0_0'),
        ('downsample2_0.ext_conv1.1', 'bn2_0_0'),
        ('downsample2_0.ext_conv1.1', 'scale2_0_0'),
        ('downsample2_0.ext_conv1.2', 'prelu2_0_0'),
        ('downsample2_0.ext_conv2.0', 'conv2_0_1'),
        ('downsample2_0.ext_conv2.1', 'bn2_0_1'),
        ('downsample2_0.ext_conv2.1', 'scale2_0_1'),
        ('downsample2_0.ext_conv2.2', 'prelu2_0_1'),
        ('downsample2_0.ext_conv3.0', 'conv2_0_2'),
        ('downsample2_0.ext_conv3.1', 'bn2_0_2'),
        ('downsample2_0.ext_conv3.1', 'scale2_0_2'),
        ('downsample2_0.ext_conv4.0', 'conv2_0_4'),
        ('downsample2_0.ext_conv4.1', 'bn2_0_4'),
        ('downsample2_0.ext_conv4.1', 'scale2_0_4'),
        ('downsample2_0.ext_conv4.2', 'prelu2_0_3'),
        ('downsample2_0.out_prelu', 'prelu2_0_4'),
        ('regular2_1.ext_conv1.0', 'conv2_1_0'),
        ('regular2_1.ext_conv1.1', 'bn2_1_0'),
        ('regular2_1.ext_conv1.1', 'scale2_1_0'),
        ('regular2_1.ext_conv1.2', 'prelu2_1_0'),
        ('regular2_1.ext_conv2.0', 'conv2_1_1'),
        ('regular2_1.ext_conv2.1', 'bn2_1_1'),
        ('regular2_1.ext_conv2.1', 'scale2_1_1'),
        ('regular2_1.ext_conv2.2', 'prelu2_1_1'),
        ('regular2_1.ext_conv3.0', 'conv2_1_2'),
        ('regular2_1.ext_conv3.1', 'bn2_1_2'),
        ('regular2_1.ext_conv3.1', 'scale2_1_2'),
        ('regular2_1.out_prelu', 'prelu2_1_4'),
        ('dilated2_2.ext_conv1.0', 'conv2_2_0'),
        ('dilated2_2.ext_conv1.1', 'bn2_2_0'),
        ('dilated2_2.ext_conv1.1', 'scale2_2_0'),
        ('dilated2_2.ext_conv1.2', 'prelu2_2_0'),
        ('dilated2_2.ext_conv2.0', 'conv2_2_1'),
        ('dilated2_2.ext_conv2.1', 'bn2_2_1'),
        ('dilated2_2.ext_conv2.1', 'scale2_2_1'),
        ('dilated2_2.ext_conv2.2', 'prelu2_2_1'),
        ('dilated2_2.ext_conv3.0', 'conv2_2_2'),
        ('dilated2_2.ext_conv3.1', 'bn2_2_2'),
        ('dilated2_2.ext_conv3.1', 'scale2_2_2'),
        ('dilated2_2.out_prelu', 'prelu2_2_4'),
        ('asymmetric2_3.ext_conv1.0', 'conv2_3_0'),
        ('asymmetric2_3.ext_conv1.1', 'bn2_3_0'),
        ('asymmetric2_3.ext_conv1.1', 'scale2_3_0'),
        ('asymmetric2_3.ext_conv1.2', 'prelu2_3_0'),
        ('asymmetric2_3.ext_conv2.0', 'conv2_3_1'),
        ('asymmetric2_3.ext_conv2.1', 'bn2_3_1'),
        ('asymmetric2_3.ext_conv2.1', 'scale2_3_1'),
        ('asymmetric2_3.ext_conv2.2', 'prelu2_3_1'),
        ('asymmetric2_3.ext_conv2.3', 'conv2_3_2'),
        ('asymmetric2_3.ext_conv2.4', 'bn2_3_2'),
        ('asymmetric2_3.ext_conv2.4', 'scale2_3_2'),
        ('asymmetric2_3.ext_conv2.5', 'prelu2_3_2'),
        ('asymmetric2_3.ext_conv3.0', 'conv2_3_3'),
        ('asymmetric2_3.ext_conv3.1', 'bn2_3_3'),
        ('asymmetric2_3.ext_conv3.1', 'scale2_3_3'),
        ('asymmetric2_3.out_prelu', 'prelu2_3_4'),
        ('dilated2_4.ext_conv1.0', 'conv2_4_0'),
        ('dilated2_4.ext_conv1.1', 'bn2_4_0'),
        ('dilated2_4.ext_conv1.1', 'scale2_4_0'),
        ('dilated2_4.ext_conv1.2', 'prelu2_4_0'),
        ('dilated2_4.ext_conv2.0', 'conv2_4_1'),
        ('dilated2_4.ext_conv2.1', 'bn2_4_1'),
        ('dilated2_4.ext_conv2.1', 'scale2_4_1'),
        ('dilated2_4.ext_conv2.2', 'prelu2_4_1'),
        ('dilated2_4.ext_conv3.0', 'conv2_4_2'),
        ('dilated2_4.ext_conv3.1', 'bn2_4_2'),
        ('dilated2_4.ext_conv3.1', 'scale2_4_2'),
        ('dilated2_4.out_prelu', 'prelu2_4_4'),
        ('regular2_5.ext_conv1.0', 'conv2_5_0'),
        ('regular2_5.ext_conv1.1', 'bn2_5_0'),
        ('regular2_5.ext_conv1.1', 'scale2_5_0'),
        ('regular2_5.ext_conv1.2', 'prelu2_5_0'),
        ('regular2_5.ext_conv2.0', 'conv2_5_1'),
        ('regular2_5.ext_conv2.1', 'bn2_5_1'),
        ('regular2_5.ext_conv2.1', 'scale2_5_1'),
        ('regular2_5.ext_conv2.2', 'prelu2_5_1'),
        ('regular2_5.ext_conv3.0', 'conv2_5_2'),
        ('regular2_5.ext_conv3.1', 'bn2_5_2'),
        ('regular2_5.ext_conv3.1', 'scale2_5_2'),
        ('regular2_5.out_prelu', 'prelu2_5_4'),
        ('dilated2_6.ext_conv1.0', 'conv2_6_0'),
        ('dilated2_6.ext_conv1.1', 'bn2_6_0'),
        ('dilated2_6.ext_conv1.1', 'scale2_6_0'),
        ('dilated2_6.ext_conv1.2', 'prelu2_6_0'),
        ('dilated2_6.ext_conv2.0', 'conv2_6_1'),
        ('dilated2_6.ext_conv2.1', 'bn2_6_1'),
        ('dilated2_6.ext_conv2.1', 'scale2_6_1'),
        ('dilated2_6.ext_conv2.2', 'prelu2_6_1'),
        ('dilated2_6.ext_conv3.0', 'conv2_6_2'),
        ('dilated2_6.ext_conv3.1', 'bn2_6_2'),
        ('dilated2_6.ext_conv3.1', 'scale2_6_2'),
        ('dilated2_6.out_prelu', 'prelu2_6_4'),
        ('asymmetric2_7.ext_conv1.0', 'conv2_7_0'),
        ('asymmetric2_7.ext_conv1.1', 'bn2_7_0'),
        ('asymmetric2_7.ext_conv1.1', 'scale2_7_0'),
        ('asymmetric2_7.ext_conv1.2', 'prelu2_7_0'),
        ('asymmetric2_7.ext_conv2.0', 'conv2_7_1'),
        ('asymmetric2_7.ext_conv2.1', 'bn2_7_1'),
        ('asymmetric2_7.ext_conv2.1', 'scale2_7_1'),
        ('asymmetric2_7.ext_conv2.2', 'prelu2_7_1'),
        ('asymmetric2_7.ext_conv2.3', 'conv2_7_2'),
        ('asymmetric2_7.ext_conv2.4', 'bn2_7_2'),
        ('asymmetric2_7.ext_conv2.4', 'scale2_7_2'),
        ('asymmetric2_7.ext_conv2.5', 'prelu2_7_2'),
        ('asymmetric2_7.ext_conv3.0', 'conv2_7_3'),
        ('asymmetric2_7.ext_conv3.1', 'bn2_7_3'),
        ('asymmetric2_7.ext_conv3.1', 'scale2_7_3'),
        ('asymmetric2_7.out_prelu', 'prelu2_7_4'),
        ('dilated2_8.ext_conv1.0', 'conv2_8_0'),
        ('dilated2_8.ext_conv1.1', 'bn2_8_0'),
        ('dilated2_8.ext_conv1.1', 'scale2_8_0'),
        ('dilated2_8.ext_conv1.2', 'prelu2_8_0'),
        ('dilated2_8.ext_conv2.0', 'conv2_8_1'),
        ('dilated2_8.ext_conv2.1', 'bn2_8_1'),
        ('dilated2_8.ext_conv2.1', 'scale2_8_1'),
        ('dilated2_8.ext_conv2.2', 'prelu2_8_1'),
        ('dilated2_8.ext_conv3.0', 'conv2_8_2'),
        ('dilated2_8.ext_conv3.1', 'bn2_8_2'),
        ('dilated2_8.ext_conv3.1', 'scale2_8_2'),
        ('dilated2_8.out_prelu', 'prelu2_8_4'),
        ('upsample4_0_1.ext_conv1.0', 'conv4_0_0'),
        ('upsample4_0_1.ext_conv1.1', 'bn4_0_0'),
        ('upsample4_0_1.ext_conv1.1', 'scale4_0_0'),
        ('upsample4_0_1.ext_conv1.2', 'prelu4_0_0'),
        ('upsample4_0_1.ext_conv2.0', 'decon4_0_1'),
        ('upsample4_0_1.ext_conv2.1', 'bn4_0_1'),
        ('upsample4_0_1.ext_conv2.1', 'scale4_0_1'),
        ('upsample4_0_1.ext_conv2.2', 'prelu4_0_1'),
        ('upsample4_0_1.ext_conv3.0', 'conv4_0_2'),
        ('upsample4_0_1.ext_conv3.1', 'bn4_0_2'),
        ('upsample4_0_1.ext_conv3.1', 'scale4_0_2'),
        ('upsample4_0_1.main_conv1.0', 'conv4_0_4'),
        ('upsample4_0_1.main_conv1.1', 'bn4_0_4'),
        ('upsample4_0_1.main_conv1.1', 'scale4_0_4'),
        ('upsample4_0_1.out_prelu', 'prelu4_0_4'),
        ('regular4_1_1.ext_conv1.0', 'conv4_1_0'),
        ('regular4_1_1.ext_conv1.1', 'bn4_1_0'),
        ('regular4_1_1.ext_conv1.1', 'scale4_1_0'),
        ('regular4_1_1.ext_conv1.2', 'prelu4_1_0'),
        ('regular4_1_1.ext_conv2.0', 'conv4_1_1'),
        ('regular4_1_1.ext_conv2.1', 'bn4_1_1'),
        ('regular4_1_1.ext_conv2.1', 'scale4_1_1'),
        ('regular4_1_1.ext_conv2.2', 'prelu4_1_1'),
        ('regular4_1_1.ext_conv3.0', 'conv4_1_2'),
        ('regular4_1_1.ext_conv3.1', 'bn4_1_2'),
        ('regular4_1_1.ext_conv3.1', 'scale4_1_2'),
        ('regular4_1_1.out_prelu', 'prelu4_1_4'),
        ('regular4_2_1.ext_conv1.0', 'conv4_2_0'),
        ('regular4_2_1.ext_conv1.1', 'bn4_2_0'),
        ('regular4_2_1.ext_conv1.1', 'scale4_2_0'),
        ('regular4_2_1.ext_conv1.2', 'prelu4_2_0'),
        ('regular4_2_1.ext_conv2.0', 'conv4_2_1'),
        ('regular4_2_1.ext_conv2.1', 'bn4_2_1'),
        ('regular4_2_1.ext_conv2.1', 'scale4_2_1'),
        ('regular4_2_1.ext_conv2.2', 'prelu4_2_1'),
        ('regular4_2_1.ext_conv3.0', 'conv4_2_2'),
        ('regular4_2_1.ext_conv3.1', 'bn4_2_2'),
        ('regular4_2_1.ext_conv3.1', 'scale4_2_2'),
        ('regular4_2_1.out_prelu', 'prelu4_2_4'),

        ('conv_out.0', 'conv_out'),
        ('conv_out.1', 'out_bn'),
        ('conv_out.1', 'out_scale'),
        ('deconv2.0', 'decon5_0_0'),
        ('deconv2.1', 'bn5_0_0'),
        ('deconv2.1', 'scale5_0_0'),
        ('deconv3.0', 'decon6_0_0'),
    ]


    # transform box_head
    modelpath="./output/belt/BeltNet.pth"
    #modelpath='./LaneNet_best.pth'#experiments_belt/
    save_path = './tools/belt_seg.caffemodel'
    pytorch_net = torch.load(modelpath, map_location='cpu')['net']
    net = caffe.Net('./tools/belt_seg.prototxt', caffe.TEST)

    caffe_params = net.params

    for torch_key, caffe_key in weight_map:
        print(torch_key)
        print(caffe_key)
        if 'conv' in caffe_key:
            print("##############################")
            save_conv(torch_key,caffe_key,pytorch_net,net)
        elif 'bn' in caffe_key:
            save_batch_norm(torch_key, caffe_key, pytorch_net, net)
        elif 'scale' in caffe_key:
            save_scale(torch_key,caffe_key,pytorch_net,net)
        elif 'decon' in caffe_key:
            print('**********************************!')
            save_deconv(torch_key,caffe_key,pytorch_net,net)
            #save_conv(torch_key,caffe_key,pytorch_net,net)
        elif 'prelu' in caffe_key:
            #print(caffe_key)
            save_prelu(torch_key,caffe_key,pytorch_net,net)
        elif 'fc' in caffe_key:
            print('***************fc*******************!')
            save_fc(torch_key,caffe_key,pytorch_net,net)
        else:
            print("Wrong type!")
    print("---------transform complete---------")

    net.save(save_path)
    print("save success!")
    #sys.exit()
    if 0:
        img = np.ones([1,3,64,128])
        img = img.astype(np.float32)
        #net2 = LANENet(input_size=(128, 64),num_classes=11)
        #net2 = LANENOVANet(input_size=(128, 64), num_classes=11, pretrained=False)
        net2 = LANENET_NOVT(input_size=(128, 64), num_classes=11)
        save_dict = torch.load(modelpath)
        net2.load_state_dict(save_dict['net'])
        net2.eval()
        net2.cuda()
        input_var = Variable(torch.FloatTensor(torch.from_numpy(img)).cuda())
        #print (net2)
        #print ("########################################################################")
        #print (net2.initial_block)
        net3=net2.initial_block
        net4=net2.downsample1_0
        net5=net2.regular1_1
        net6=net2.regular1_2
        net7=net2.regular1_3
        net8=net2.regular1_4
        net9=net2.downsample2_0
        net10=net2.regular2_1
        net11=net2.regular2_2
        # x = self.regular2_3(x)
        # x = self.regular2_4(x)
        # x = self.regular2_5(x)
        # x = self.regular2_6(x)
        # x = self.regular2_7(x)
        # x = self.regular2_8(x)
        net12=net2.regular2_3
        net13=net2.regular2_4
        net14=net2.regular2_5
        net15=net2.regular2_6
        net16=net2.regular2_7
        net17=net2.regular2_8
        # x = self.upsample4_0_1(x, max_indices2_0)
        # x = self.regular4_1_1(x)
        # x = self.regular4_2_1(x)
        net18=net2.deconv1
        net19=net2.regular4_1_1
        net20=net2.regular4_2_1
        net21=net2.conv_out
        net22=net2.deconv2
        net23=net2.deconv3
        # print (net21[1])
        # sys.exit()
        seg_pred1, _ = net2(input_var)[:2]
        seg_pred= net3(input_var)
        seg_pred,_=net4(seg_pred)
        seg_pred=net5(seg_pred)
        seg_pred=net6(seg_pred)
        seg_pred=net7(seg_pred)
        seg_pred=net8(seg_pred)
        seg_pred, max_indices2_0=net9(seg_pred)
        seg_pred=net10(seg_pred)
        seg_pred=net11(seg_pred)
        seg_pred=net12(seg_pred)
        seg_pred=net13(seg_pred)
        seg_pred=net14(seg_pred)
        seg_pred=net15(seg_pred)
        seg_pred=net16(seg_pred)
        seg_pred=net17(seg_pred)
        seg_pred=net18(seg_pred)
        seg_pred=net19(seg_pred)
        seg_pred=net20(seg_pred)
        seg_pred=net21(seg_pred)
        seg_pred=net22(seg_pred)
        seg_pred=net23(seg_pred)
        #seg_pred=net21[1](seg_pred)

        seg_pred1 = seg_pred1.detach().cpu().numpy()
        #print (seg_pred.shape)
        #sys.exit()
        timg = np.ones([3,64,128])
        timg = timg.astype(np.float32)
        net.blobs['data'].data[...] = timg
        outc=net.forward()#['cnn_layer']
        caffeout=net.blobs['decon6_0_0'].data
        out2=seg_pred1.reshape(-1)
        caffeout=caffeout.reshape(-1)
        print(out2.shape,caffeout.shape)
        for i in range(out2.shape[0]):
            if(abs(out2[i]-caffeout[i])>0.01):
                print("error", abs(out2[i]-caffeout[i]) )