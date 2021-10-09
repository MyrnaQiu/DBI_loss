#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:49:43 2018

@author: myrna
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks_other import init_weights
from .attention import *
import importlib

class dethwise_block(nn.Module):
    #expand + depthwise + pointwise
    #att = None,se,cbam,hwd
    #nonlinear = Relu,LeakyRelu
    def __init__(self,  in_size, out_size, expand_size, nolinear, attmodule, kernel_size, padding_size, stride):
        super(dethwise_block, self).__init__()
        self.stride = stride
        self.att = attmodule
        self.padding = padding_size

        self.conv1 = nn.Conv3d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(expand_size)
        #self.bn1 = nn.GroupNorm(num_channels=expand_size,num_groups=2)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv3d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2,kernel_size[1]//2,kernel_size[2]//2), groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm3d(expand_size)
        #self.bn2 = nn.GroupNorm(num_channels=expand_size, num_groups=2)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv3d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_size)
        #self.bn3 = nn.GroupNorm(num_channels=out_size, num_groups=2)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_size),
                #nn.GroupNorm(num_channels=out_size, num_groups=2),
            )
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.att != None:
            out = self.att(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return outfz


class basconv_block(nn.Module):
    def __init__(self, in_size, out_size, nolinear, attmodule, is_res, kernel_size, padding_size, stride):
        super(basconv_block, self).__init__()

        self.nolinear1 = nolinear
        self.nolinear2 = nolinear
        self.att = attmodule
        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding_size, bias=True),
                                   nn.BatchNorm3d(out_size),)
                                   #nn.GroupNorm(num_channels=out_size, num_groups=2))
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, stride, padding_size, bias=True),
                                   nn.BatchNorm3d(out_size),)
                                   #nn.GroupNorm(num_channels=out_size, num_groups=2))
        self.is_res = is_res
        if self.is_res:
            self.reconv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, 1, padding_size, bias=True),
                                        nn.BatchNorm3d(out_size), )

    def forward(self, inputs):
        outputs = self.nolinear1(self.conv1(inputs))
        outputs = self.nolinear2(self.conv2(outputs))

        if self.att != None:
            outputs = self.att(outputs)

        if self.is_res:
            residual = self.reconv(outputs)
            outputs += residual

        return outputs

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_dethwise, expand_size, nolinear, attmodule, is_res, kernel_size, padding_size, stride):
        super(UnetConv3, self).__init__()

        if is_dethwise:
            #in_size, out_size, kernel_size, expand_size, padding_size, stride, nolinear, attmodule
            self.conv1 = dethwise_block(in_size, out_size, expand_size, nolinear, attmodule, kernel_size, padding_size, stride)
        else:
            self.conv1 = basconv_block(in_size, out_size, nolinear, attmodule, is_res, kernel_size, padding_size, stride)


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_dethwise, expand_size, nolinear, attmodule, is_res, kernel_size, padding_size, stride,scale_factor):
        super(UnetUp3, self).__init__()

        self.scale_factor = scale_factor
        self.conv = UnetConv3(in_size+out_size, out_size, is_dethwise, expand_size, nolinear, attmodule, is_res, kernel_size, padding_size, stride)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = nn.functional.interpolate(input=inputs2, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        cat_outputs = torch.cat([outputs1, outputs2], 1)  #[2,192,16,16,6]
        return self.conv(cat_outputs)


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True, bias = True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:       # k=1 stride=1, padding=0
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0),bias=bias),
                                       nn.BatchNorm3d(out_size),
                                       #nn.GroupNorm(num_channels=out_size, num_groups=2),
                                       nn.LeakyReLU(inplace = True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0),bias=bias),
                                       nn.LeakyReLU(inplace = True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
    
    
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor,bias=True):
        super(UnetDsv3, self).__init__()
        self.scale_factor = scale_factor
        self.dsv = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0,bias=bias)

    def forward(self, input):
        input1 = self.dsv(input)
        input2 = nn.functional.interpolate(input=input1, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        return input2
    

    
class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = nn.functional.interpolate(input=(self.outputs[index]).data(), size=newsize[2:], mode='bilinear',align_corners=True)
        else:
            self.outputs = nn.functional.interpolate(input=(self.outputs).data(), size=newsize[2:], mode='bilinear',align_corners=True)

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=6):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool3d(x, kernel_size=x.size()[2:5]) #[2, 256+512, 1, 1, 1]]
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y)) #[2, 1, 1, 1, 128]
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y

class UnetUp3_SqEx(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False):
        super(UnetUp3_SqEx, self).__init__()
        if is_deconv:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size)
            self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)) #[2,512,16,16,6]
        else:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size,kernel_size=(3,3,3), padding_size=(1,1,1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = nn.functional.interpolat(input=inputs2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        offset = outputs2.size()[2] - inputs1.size()[2]  # 0
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        concat = torch.cat([outputs1, outputs2], 1)
        gated  = self.sqex(concat)
        return self.conv(gated)
    
class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    ''
class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv3d(in_channels, 128, kernel_size=1)

        self.branch5x5_1 = BasicConv3d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv3d(64, 128, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv3d(in_channels, 128, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv3d(128, 192, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv3d(192, 192, kernel_size=3, padding=1)

        self.branch_pool = BasicConv3d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
