
import sys
import torch
import numpy as np
import os

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def save_conv(torch_name, caffe_name, weights, net):

    if caffe_name == 'conv8_0_0':
        conv_key = '{}.weight'.format(torch_name)
        bias_key = '{}.bias'.format(torch_name)
        conv_weight = weights[conv_key].cpu().numpy()
        conv_bias = weights[bias_key].cpu().numpy()
        net.params[caffe_name][0].data[...] = conv_weight
        net.params[caffe_name][1].data[...] = conv_bias
    else:
        conv_key = '{}.weight'.format(torch_name)
        print(conv_key)
        conv_weight = weights[conv_key].cpu().numpy()
        net.params[caffe_name][0].data[...] = conv_weight
        #print(conv_weight)

    print("conv: add weight of {} success!".format(caffe_name))


def save_deconv(torch_name, caffe_name, weights, net):
    deconv_key = '{}.weight'.format(torch_name)
    bias_key = '{}.bias'.format(torch_name)
    deconv_weight = weights[deconv_key].cpu().numpy()
    deconv_bias = weights[bias_key].cpu().numpy()
    net.params[caffe_name][0].data[...] = deconv_weight
    net.params[caffe_name][1].data[...] = deconv_bias
    print(deconv_weight)
    print(deconv_bias)
    print("deconv: add weight of {} success!".format(caffe_name))


def save_batch_norm(torch_name, caffe_name, weights, net):
    mean_key = '{}.running_mean'.format(torch_name)
    var_key = '{}.running_var'.format(torch_name)
    #print(mean_key)
    #print(var_key)
    running_mean = weights[mean_key].cpu().numpy()
    running_var = weights[var_key].cpu().numpy()
    print(running_mean)
    print(running_var)
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
    print(scale_weight)
    net.params[caffe_name][0].data[...] = scale_weight
    net.params[caffe_name][1].data[...] = scale_bias
    print("add weight of {} success!".format(caffe_name))


def save_prelu(torch_name, caffe_name, weights, net):
    weight_key = '{}.weight'.format(torch_name)
    #print(weight_key)
    prelu_weight = weights[weight_key].cpu().numpy()
    #print(prelu_weight)
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
        ('initial_block_new.main_branch', 'conv0_1'),
        ('initial_block_new.batch_norm', 'bn0_1'),
        ('initial_block_new.batch_norm', 'scale0_1'),
        #('initial_block.out_prelu', 'prelu0_1'),
        ('downsample1_0.ext_conv1.0', 'conv1_0_0'),
        ('downsample1_0.ext_conv1.1', 'bn1_0_0'),
        ('downsample1_0.ext_conv1.1', 'scale1_0_0'),
        #('downsample1_0.ext_conv1.2', 'prelu1_0_0'),
        ('downsample1_0.ext_conv2.0', 'conv1_0_1'),
        ('downsample1_0.ext_conv2.1', 'bn1_0_1'),
        ('downsample1_0.ext_conv2.1', 'scale1_0_1'),
        #('downsample1_0.ext_conv2.2', 'prelu1_0_1'),
        ('downsample1_0.ext_conv3.0', 'conv1_0_2'),
        ('downsample1_0.ext_conv3.1', 'bn1_0_2'),
        ('downsample1_0.ext_conv3.1', 'scale1_0_2'),
        ('downsample1_0.ext_conv4.0', 'conv1_0_4'),
        ('downsample1_0.ext_conv4.1', 'bn1_0_4'),
        ('downsample1_0.ext_conv4.1', 'scale1_0_4'),
        #('downsample1_0.ext_conv4.2', 'prelu1_0_3'),
        #('downsample1_0.out_prelu', 'prelu1_0_4'),
        ('regular1_1.ext_conv1.0', 'conv1_1_0'),
        ('regular1_1.ext_conv1.1', 'bn1_1_0'),
        ('regular1_1.ext_conv1.1', 'scale1_1_0'),
        #('regular1_1.ext_conv1.2', 'prelu1_1_0'),
        ('regular1_1.ext_conv2.0', 'conv1_1_1'),
        ('regular1_1.ext_conv2.1', 'bn1_1_1'),
        ('regular1_1.ext_conv2.1', 'scale1_1_1'),
        #('regular1_1.ext_conv2.2', 'prelu1_1_1'),
        ('regular1_1.ext_conv3.0', 'conv1_1_2'),
        ('regular1_1.ext_conv3.1', 'bn1_1_2'),
        ('regular1_1.ext_conv3.1', 'scale1_1_2'),
        #('regular1_1.out_prelu', 'prelu1_1_4'),
        ('regular1_2.ext_conv1.0', 'conv1_2_0'),
        ('regular1_2.ext_conv1.1', 'bn1_2_0'),
        ('regular1_2.ext_conv1.1', 'scale1_2_0'),
        #('regular1_2.ext_conv1.2', 'prelu1_2_0'),
        ('regular1_2.ext_conv2.0', 'conv1_2_1'),
        ('regular1_2.ext_conv2.1', 'bn1_2_1'),
        ('regular1_2.ext_conv2.1', 'scale1_2_1'),
        #('regular1_2.ext_conv2.2', 'prelu1_2_1'),
        ('regular1_2.ext_conv3.0', 'conv1_2_2'),
        ('regular1_2.ext_conv3.1', 'bn1_2_2'),
        ('regular1_2.ext_conv3.1', 'scale1_2_2'),
        #('regular1_2.ext_conv3.2', 'prelu1_2_2'),
        #('regular1_2.out_prelu', 'prelu1_2_4'),
        ('regular1_3.ext_conv1.0', 'conv1_3_0'),
        ('regular1_3.ext_conv1.1', 'bn1_3_0'),
        ('regular1_3.ext_conv1.1', 'scale1_3_0'),
        #('regular1_3.ext_conv1.2', 'prelu1_3_0'),
        ('regular1_3.ext_conv2.0', 'conv1_3_1'),
        ('regular1_3.ext_conv2.1', 'bn1_3_1'),
        ('regular1_3.ext_conv2.1', 'scale1_3_1'),
        #('regular1_3.ext_conv2.2', 'prelu1_3_1'),
        ('regular1_3.ext_conv3.0', 'conv1_3_2'),
        ('regular1_3.ext_conv3.1', 'bn1_3_2'),
        ('regular1_3.ext_conv3.1', 'scale1_3_2'),
        #('regular1_3.ext_conv3.2', 'prelu1_3_2'),
        #('regular1_3.out_prelu', 'prelu1_3_4'),
        ('regular1_4.ext_conv1.0', 'conv1_4_0'),
        ('regular1_4.ext_conv1.1', 'bn1_4_0'),
        ('regular1_4.ext_conv1.1', 'scale1_4_0'),
        #('regular1_4.ext_conv1.2', 'prelu1_4_0'),
        ('regular1_4.ext_conv2.0', 'conv1_4_1'),
        ('regular1_4.ext_conv2.1', 'bn1_4_1'),
        ('regular1_4.ext_conv2.1', 'scale1_4_1'),
        #('regular1_4.ext_conv2.2', 'prelu1_4_1'),
        ('regular1_4.ext_conv3.0', 'conv1_4_2'),
        ('regular1_4.ext_conv3.1', 'bn1_4_2'),
        ('regular1_4.ext_conv3.1', 'scale1_4_2'),
        #('regular1_4.ext_conv3.2', 'prelu1_4_2'),
        #('regular1_4.out_prelu', 'prelu1_4_4'),
        ('downsample2_0.ext_conv1.0', 'conv2_0_0'),
        ('downsample2_0.ext_conv1.1', 'bn2_0_0'),
        ('downsample2_0.ext_conv1.1', 'scale2_0_0'),
        #('downsample2_0.ext_conv1.2', 'prelu2_0_0'),
        ('downsample2_0.ext_conv2.0', 'conv2_0_1'),
        ('downsample2_0.ext_conv2.1', 'bn2_0_1'),
        ('downsample2_0.ext_conv2.1', 'scale2_0_1'),
        #('downsample2_0.ext_conv2.2', 'prelu2_0_1'),
        ('downsample2_0.ext_conv3.0', 'conv2_0_2'),
        ('downsample2_0.ext_conv3.1', 'bn2_0_2'),
        ('downsample2_0.ext_conv3.1', 'scale2_0_2'),
        #('downsample2_0.ext_conv3.2', 'prelu2_0_2'),
        ('downsample2_0.ext_conv4.0', 'conv2_0_4'),
        ('downsample2_0.ext_conv4.1', 'bn2_0_4'),
        ('downsample2_0.ext_conv4.1', 'scale2_0_4'),
        #('downsample2_0.ext_conv4.2', 'prelu2_0_3'),
        #('downsample2_0.out_prelu', 'prelu2_0_4'),
        ('regular2_1.ext_conv1.0', 'conv2_1_0'),
        ('regular2_1.ext_conv1.1', 'bn2_1_0'),
        ('regular2_1.ext_conv1.1', 'scale2_1_0'),
        #('regular2_1.ext_conv1.2', 'prelu2_1_0'),
        ('regular2_1.ext_conv2.0', 'conv2_1_1'),
        ('regular2_1.ext_conv2.1', 'bn2_1_1'),
        ('regular2_1.ext_conv2.1', 'scale2_1_1'),
        #('regular2_1.ext_conv2.2', 'prelu2_1_1'),
        ('regular2_1.ext_conv3.0', 'conv2_1_2'),
        ('regular2_1.ext_conv3.1', 'bn2_1_2'),
        ('regular2_1.ext_conv3.1', 'scale2_1_2'),
        #('regular2_1.ext_conv3.2', 'prelu2_1_2'),
        #('regular2_1.out_prelu', 'prelu2_1_4'),
        ('regular2_2.ext_conv1.0', 'conv2_2_0'),
        ('regular2_2.ext_conv1.1', 'bn2_2_0'),
        ('regular2_2.ext_conv1.1', 'scale2_2_0'),
        #('regular2_2.ext_conv1.2', 'prelu2_2_0'),
        ('regular2_2.ext_conv2.0', 'conv2_2_1'),
        ('regular2_2.ext_conv2.1', 'bn2_2_1'),
        ('regular2_2.ext_conv2.1', 'scale2_2_1'),
        #('regular2_2.ext_conv2.2', 'prelu2_2_1'),
        ('regular2_2.ext_conv3.0', 'conv2_2_2'),
        ('regular2_2.ext_conv3.1', 'bn2_2_2'),
        ('regular2_2.ext_conv3.1', 'scale2_2_2'),
        #('dilated2_2.ext_conv3.2', 'prelu2_2_2'),
        #('regular2_2.out_prelu', 'prelu2_2_4'),
        ('regular2_3.ext_conv1.0', 'conv2_3_0'),
        ('regular2_3.ext_conv1.1', 'bn2_3_0'),
        ('regular2_3.ext_conv1.1', 'scale2_3_0'),
        #('regular2_3.ext_conv1.2', 'prelu2_3_0'),
        ('regular2_3.ext_conv2.0', 'conv2_3_1'),
        ('regular2_3.ext_conv2.1', 'bn2_3_1'),
        ('regular2_3.ext_conv2.1', 'scale2_3_1'),
        #('regular2_3.ext_conv2.2', 'prelu2_3_1'),
        ('regular2_3.ext_conv3.0', 'conv2_3_3'),
        ('regular2_3.ext_conv3.1', 'bn2_3_3'),
        ('regular2_3.ext_conv3.1', 'scale2_3_3'),
        #('asymmetric2_3.ext_conv3.2', 'prelu2_3_3'),
        #('regular2_3.out_prelu', 'prelu2_3_4'),
        ('regular2_4.ext_conv1.0', 'conv2_4_0'),
        ('regular2_4.ext_conv1.1', 'bn2_4_0'),
        ('regular2_4.ext_conv1.1', 'scale2_4_0'),
        #('regular2_4.ext_conv1.2', 'prelu2_4_0'),
        ('regular2_4.ext_conv2.0', 'conv2_4_1'),
        ('regular2_4.ext_conv2.1', 'bn2_4_1'),
        ('regular2_4.ext_conv2.1', 'scale2_4_1'),
        #('regular2_4.ext_conv2.2', 'prelu2_4_1'),
        ('regular2_4.ext_conv3.0', 'conv2_4_2'),
        ('regular2_4.ext_conv3.1', 'bn2_4_2'),
        ('regular2_4.ext_conv3.1', 'scale2_4_2'),
        #('dilated2_4.ext_conv3.2', 'prelu2_4_2'),
        #('regular2_4.out_prelu', 'prelu2_4_4'),
        ('regular2_5.ext_conv1.0', 'conv2_5_0'),
        ('regular2_5.ext_conv1.1', 'bn2_5_0'),
        ('regular2_5.ext_conv1.1', 'scale2_5_0'),
        #('regular2_5.ext_conv1.2', 'prelu2_5_0'),
        ('regular2_5.ext_conv2.0', 'conv2_5_1'),
        ('regular2_5.ext_conv2.1', 'bn2_5_1'),
        ('regular2_5.ext_conv2.1', 'scale2_5_1'),
        #('regular2_5.ext_conv2.2', 'prelu2_5_1'),
        ('regular2_5.ext_conv3.0', 'conv2_5_2'),
        ('regular2_5.ext_conv3.1', 'bn2_5_2'),
        ('regular2_5.ext_conv3.1', 'scale2_5_2'),
        #('regular2_5.ext_conv3.2', 'prelu2_5_2'),
        #('regular2_5.out_prelu', 'prelu2_5_4'),
        ('regular2_6.ext_conv1.0', 'conv2_6_0'),
        ('regular2_6.ext_conv1.1', 'bn2_6_0'),
        ('regular2_6.ext_conv1.1', 'scale2_6_0'),
        #('regular2_6.ext_conv1.2', 'prelu2_6_0'),
        ('regular2_6.ext_conv2.0', 'conv2_6_1'),
        ('regular2_6.ext_conv2.1', 'bn2_6_1'),
        ('regular2_6.ext_conv2.1', 'scale2_6_1'),
        #('regular2_6.ext_conv2.2', 'prelu2_6_1'),
        ('regular2_6.ext_conv3.0', 'conv2_6_2'),
        ('regular2_6.ext_conv3.1', 'bn2_6_2'),
        ('regular2_6.ext_conv3.1', 'scale2_6_2'),
        #('dilated2_6.ext_conv3.2', 'prelu2_6_2'),
        #('regular2_6.out_prelu', 'prelu2_6_4'),
        ('regular2_7.ext_conv1.0', 'conv2_7_0'),
        ('regular2_7.ext_conv1.1', 'bn2_7_0'),
        ('regular2_7.ext_conv1.1', 'scale2_7_0'),
        #('regular2_7.ext_conv1.2', 'prelu2_7_0'),
        ('regular2_7.ext_conv2.0', 'conv2_7_1'),
        ('regular2_7.ext_conv2.1', 'bn2_7_1'),
        ('regular2_7.ext_conv2.1', 'scale2_7_1'),
        #('regular2_7.ext_conv2.2', 'prelu2_7_1'),
        ('regular2_7.ext_conv3.0', 'conv2_7_3'),
        ('regular2_7.ext_conv3.1', 'bn2_7_3'),
        ('regular2_7.ext_conv3.1', 'scale2_7_3'),
        #('asymmetric2_7.ext_conv3.2', 'prelu2_7_3'),
        #('regular2_7.out_prelu', 'prelu2_7_4'),
        ('regular2_8.ext_conv1.0', 'conv2_8_0'),
        ('regular2_8.ext_conv1.1', 'bn2_8_0'),
        ('regular2_8.ext_conv1.1', 'scale2_8_0'),
        #('regular2_8.ext_conv1.2', 'prelu2_8_0'),
        ('regular2_8.ext_conv2.0', 'conv2_8_1'),
        ('regular2_8.ext_conv2.1', 'bn2_8_1'),
        ('regular2_8.ext_conv2.1', 'scale2_8_1'),
        #('regular2_8.ext_conv2.2', 'prelu2_8_1'),
        ('regular2_8.ext_conv3.0', 'conv2_8_2'),
        ('regular2_8.ext_conv3.1', 'bn2_8_2'),
        ('regular2_8.ext_conv3.1', 'scale2_8_2'),
        #('dilated2_8.ext_conv3.2', 'prelu2_8_2'),
        ('deconv1.0', 'decon4_0_0'),
        ('deconv1.1', 'bn4_0_0'),
        ('deconv1.1', 'scale4_0_0'),        
        ('deconv2.0', 'decon5_0_0'),
        ('deconv2.1', 'bn5_0_0'),
        ('deconv2.1', 'scale5_0_0'),
        ('deconv3.0', 'decon5_0_1'),


        ('downsample3_0.ext_conv1.0', 'conv3_0_0'),
        ('downsample3_0.ext_conv1.1', 'bn3_0_0'),
        ('downsample3_0.ext_conv1.1', 'scale3_0_0'),
        ('downsample3_0.ext_conv2.0', 'conv3_0_1'),
        ('downsample3_0.ext_conv2.1', 'bn3_0_1'),
        ('downsample3_0.ext_conv2.1', 'scale3_0_1'),
        ('downsample3_0.ext_conv3.0', 'conv3_0_2'),
        ('downsample3_0.ext_conv3.1', 'bn3_0_2'),
        ('downsample3_0.ext_conv3.1', 'scale3_0_2'),
        ('downsample3_0.ext_conv4.0', 'conv3_0_4'),
        ('downsample3_0.ext_conv4.1', 'bn3_0_4'),
        ('downsample3_0.ext_conv4.1', 'scale3_0_4'),
        ('regular3_1.ext_conv1.0', 'conv3_1_0'),
        ('regular3_1.ext_conv1.1', 'bn3_1_0'),
        ('regular3_1.ext_conv1.1', 'scale3_1_0'),
        ('regular3_1.ext_conv2.0', 'conv3_1_1'),
        ('regular3_1.ext_conv2.1', 'bn3_1_1'),
        ('regular3_1.ext_conv2.1', 'scale3_1_1'),
        ('regular3_1.ext_conv3.0', 'conv3_1_2'),
        ('regular3_1.ext_conv3.1', 'bn3_1_2'),
        ('regular3_1.ext_conv3.1', 'scale3_1_2'),
        ('regular3_2.ext_conv1.0', 'conv3_2_0'),
        ('regular3_2.ext_conv1.1', 'bn3_2_0'),
        ('regular3_2.ext_conv1.1', 'scale3_2_0'),
        ('regular3_2.ext_conv2.0', 'conv3_2_1'),
        ('regular3_2.ext_conv2.1', 'bn3_2_1'),
        ('regular3_2.ext_conv2.1', 'scale3_2_1'),
        ('regular3_2.ext_conv3.0', 'conv3_2_2'),
        ('regular3_2.ext_conv3.1', 'bn3_2_2'),
        ('regular3_2.ext_conv3.1', 'scale3_2_2'),
        ('regular3_3.ext_conv1.0', 'conv3_3_0'),
        ('regular3_3.ext_conv1.1', 'bn3_3_0'),
        ('regular3_3.ext_conv1.1', 'scale3_3_0'),
        ('regular3_3.ext_conv2.0', 'conv3_3_1'),
        ('regular3_3.ext_conv2.1', 'bn3_3_1'),
        ('regular3_3.ext_conv2.1', 'scale3_3_1'),
        ('regular3_3.ext_conv3.0', 'conv3_3_3'),
        ('regular3_3.ext_conv3.1', 'bn3_3_3'),
        ('regular3_3.ext_conv3.1', 'scale3_3_3'),
        ('regular3_4.ext_conv1.0', 'conv3_4_0'),
        ('regular3_4.ext_conv1.1', 'bn3_4_0'),
        ('regular3_4.ext_conv1.1', 'scale3_4_0'),
        ('regular3_4.ext_conv2.0', 'conv3_4_1'),
        ('regular3_4.ext_conv2.1', 'bn3_4_1'),
        ('regular3_4.ext_conv2.1', 'scale3_4_1'),
        ('regular3_4.ext_conv3.0', 'conv3_4_3'),
        ('regular3_4.ext_conv3.1', 'bn3_4_3'),
        ('regular3_4.ext_conv3.1', 'scale3_4_3'),


        ('downsample4_0.ext_conv1.0', 'conv6_0_0'),
        ('downsample4_0.ext_conv1.1', 'bn6_0_0'),
        ('downsample4_0.ext_conv1.1', 'scale6_0_0'),
        ('downsample4_0.ext_conv2.0', 'conv6_0_1'),
        ('downsample4_0.ext_conv2.1', 'bn6_0_1'),
        ('downsample4_0.ext_conv2.1', 'scale6_0_1'),
        ('downsample4_0.ext_conv3.0', 'conv6_0_2'),
        ('downsample4_0.ext_conv3.1', 'bn6_0_2'),
        ('downsample4_0.ext_conv3.1', 'scale6_0_2'),
        ('downsample4_0.ext_conv4.0', 'conv6_0_4'),
        ('downsample4_0.ext_conv4.1', 'bn6_0_4'),
        ('downsample4_0.ext_conv4.1', 'scale6_0_4'),
        ('regular4_1.ext_conv1.0', 'conv6_1_0'),
        ('regular4_1.ext_conv1.1', 'bn6_1_0'),
        ('regular4_1.ext_conv1.1', 'scale6_1_0'),
        ('regular4_1.ext_conv2.0', 'conv6_1_1'),
        ('regular4_1.ext_conv2.1', 'bn6_1_1'),
        ('regular4_1.ext_conv2.1', 'scale6_1_1'),
        ('regular4_1.ext_conv3.0', 'conv6_1_2'),
        ('regular4_1.ext_conv3.1', 'bn6_1_2'),
        ('regular4_1.ext_conv3.1', 'scale6_1_2'),
        ('regular4_2.ext_conv1.0', 'conv6_2_0'),
        ('regular4_2.ext_conv1.1', 'bn6_2_0'),
        ('regular4_2.ext_conv1.1', 'scale6_2_0'),
        ('regular4_2.ext_conv2.0', 'conv6_2_1'),
        ('regular4_2.ext_conv2.1', 'bn6_2_1'),
        ('regular4_2.ext_conv2.1', 'scale6_2_1'),
        ('regular4_2.ext_conv3.0', 'conv6_2_2'),
        ('regular4_2.ext_conv3.1', 'bn6_2_2'),
        ('regular4_2.ext_conv3.1', 'scale6_2_2'),
        ('regular4_3.ext_conv1.0', 'conv6_3_0'),
        ('regular4_3.ext_conv1.1', 'bn6_3_0'),
        ('regular4_3.ext_conv1.1', 'scale6_3_0'),
        ('regular4_3.ext_conv2.0', 'conv6_3_1'),
        ('regular4_3.ext_conv2.1', 'bn6_3_1'),
        ('regular4_3.ext_conv2.1', 'scale6_3_1'),
        ('regular4_3.ext_conv3.0', 'conv6_3_3'),
        ('regular4_3.ext_conv3.1', 'bn6_3_3'),
        ('regular4_3.ext_conv3.1', 'scale6_3_3'),
        ('regular4_4.ext_conv1.0', 'conv6_4_0'),
        ('regular4_4.ext_conv1.1', 'bn6_4_0'),
        ('regular4_4.ext_conv1.1', 'scale6_4_0'),
        ('regular4_4.ext_conv2.0', 'conv6_4_1'),
        ('regular4_4.ext_conv2.1', 'bn6_4_1'),
        ('regular4_4.ext_conv2.1', 'scale6_4_1'),
        ('regular4_4.ext_conv3.0', 'conv6_4_3'),
        ('regular4_4.ext_conv3.1', 'bn6_4_3'),
        ('regular4_4.ext_conv3.1', 'scale6_4_3'),


        ('downsample5_0.ext_conv1.0', 'conv7_0_0'),
        ('downsample5_0.ext_conv1.1', 'bn7_0_0'),
        ('downsample5_0.ext_conv1.1', 'scale7_0_0'),
        ('downsample5_0.ext_conv2.0', 'conv7_0_1'),
        ('downsample5_0.ext_conv2.1', 'bn7_0_1'),
        ('downsample5_0.ext_conv2.1', 'scale7_0_1'),
        ('downsample5_0.ext_conv3.0', 'conv7_0_2'),
        ('downsample5_0.ext_conv3.1', 'bn7_0_2'),
        ('downsample5_0.ext_conv3.1', 'scale7_0_2'),
        ('downsample5_0.ext_conv4.0', 'conv7_0_4'),
        ('downsample5_0.ext_conv4.1', 'bn7_0_4'),
        ('downsample5_0.ext_conv4.1', 'scale7_0_4'),
        ('regular5_1.ext_conv1.0', 'conv7_1_0'),
        ('regular5_1.ext_conv1.1', 'bn7_1_0'),
        ('regular5_1.ext_conv1.1', 'scale7_1_0'),
        ('regular5_1.ext_conv2.0', 'conv7_1_1'),
        ('regular5_1.ext_conv2.1', 'bn7_1_1'),
        ('regular5_1.ext_conv2.1', 'scale7_1_1'),
        ('regular5_1.ext_conv3.0', 'conv7_1_2'),
        ('regular5_1.ext_conv3.1', 'bn7_1_2'),
        ('regular5_1.ext_conv3.1', 'scale7_1_2'),
        ('regular5_2.ext_conv1.0', 'conv7_2_0'),
        ('regular5_2.ext_conv1.1', 'bn7_2_0'),
        ('regular5_2.ext_conv1.1', 'scale7_2_0'),
        ('regular5_2.ext_conv2.0', 'conv7_2_1'),
        ('regular5_2.ext_conv2.1', 'bn7_2_1'),
        ('regular5_2.ext_conv2.1', 'scale7_2_1'),
        ('regular5_2.ext_conv3.0', 'conv7_2_2'),
        ('regular5_2.ext_conv3.1', 'bn7_2_2'),
        ('regular5_2.ext_conv3.1', 'scale7_2_2'),
        ('regular5_3.ext_conv1.0', 'conv7_3_0'),
        ('regular5_3.ext_conv1.1', 'bn7_3_0'),
        ('regular5_3.ext_conv1.1', 'scale7_3_0'),
        ('regular5_3.ext_conv2.0', 'conv7_3_1'),
        ('regular5_3.ext_conv2.1', 'bn7_3_1'),
        ('regular5_3.ext_conv2.1', 'scale7_3_1'),
        ('regular5_3.ext_conv3.0', 'conv7_3_3'),
        ('regular5_3.ext_conv3.1', 'bn7_3_3'),
        ('regular5_3.ext_conv3.1', 'scale7_3_3'),
        ('regular5_4.ext_conv1.0', 'conv7_4_0'),
        ('regular5_4.ext_conv1.1', 'bn7_4_0'),
        ('regular5_4.ext_conv1.1', 'scale7_4_0'),
        ('regular5_4.ext_conv2.0', 'conv7_4_1'),
        ('regular5_4.ext_conv2.1', 'bn7_4_1'),
        ('regular5_4.ext_conv2.1', 'scale7_4_1'),
        ('regular5_4.ext_conv3.0', 'conv7_4_3'),
        ('regular5_4.ext_conv3.1', 'bn7_4_3'),
        ('regular5_4.ext_conv3.1', 'scale7_4_3'),

        ('deconv4.0', 'conv8_0_0'),
    ]

    # pytorch_net = torch.load("/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_1117/LaneNetNova2Head.pth", map_location='cpu')['net']
    # save_path = '/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_1117/LaneNetNova_seg_city_color_class_zd_1117.caffemodel'
    # net = caffe.Net('./tools/prototxt/LaneNetNovaHeadColorClass_class_15.prototxt', caffe.TEST)

    pytorch_net = torch.load("/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_1216/LaneNetNova2Head.pth", map_location='cpu')['net']
    save_path = '/mnt/huanyuan/model/image/lpr/zd/seg_city_color_class_zd_1216/LaneNetNova_seg_city_color_class_zd_1216.caffemodel'
    net = caffe.Net('./tools/prototxt/LaneNetNovaHeadColorClass_class_17.prototxt', caffe.TEST)

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