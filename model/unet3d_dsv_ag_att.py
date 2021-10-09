#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:57:41 2019

@author: myrna
"""
import torch.nn as nn
from .utils import *
import torch.nn.functional as F
from .networks_other import init_weights
from .grid_attention_layer import GridAttentionBlock3D
from .attention import *

class unet3d_dsv_ag_att(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, in_channels=1, attention_dsample=(2, 2, 2), is_pooling=True,
                 is_dethwise=False, attmodule = None, is_res=False):

        super(unet3d_dsv_ag_att, self).__init__()
        self.in_channels = in_channels
        self.is_pooling = is_pooling
        self.is_dethwise = is_dethwise
        self.attmodule = attmodule
        self.is_res = is_res
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]  #16 32 64 128 256 512

        if self.attmodule == 'SE':
            self.att0 = SeModule(filters[0])
            self.att1 = SeModule(filters[1])
            self.att2 = SeModule(filters[2])
            self.att3 = SeModule(filters[3])
            self.att4 = SeModule(filters[4])
            self.att5 = SeModule(filters[5])
        elif self.attmodule == 'CBAM':
            #self.att0 = CBAM(filters[0])
            self.att1 = CBAM(filters[1])
            self.att2 = CBAM(filters[2])
            self.att3 = CBAM(filters[3])
            self.att4 = CBAM(filters[4])
            self.att5 = CBAM(filters[5])
        else:
            self.att0 = None
            self.att1 = None
            self.att2 = None
            self.att3 = None
            self.att4 = None
            self.att5 = None


        # downsampling

        self.conv0 = UnetConv3(self.in_channels, filters[0], self.is_dethwise, 2 * self.in_channels,
                               nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1),
                               stride=(1, 1, 1))
        self.conv1 = UnetConv3(filters[0], filters[1], self.is_dethwise, 2 * filters[0],
                               nn.LeakyReLU(inplace=True), self.att1, self.is_res, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1),
                               stride=(1, 1, 1))
        self.conv2 = UnetConv3(filters[1], filters[2], self.is_dethwise, 2 * filters[1],
                               nn.LeakyReLU(inplace=True), self.att2, self.is_res, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1),
                               stride=(1, 1, 1))
        self.conv3 = UnetConv3(filters[2], filters[3], self.is_dethwise, 2 * filters[2],
                               nn.LeakyReLU(inplace=True), self.att3, self.is_res, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1),
                               stride=(1, 1, 1))
        self.conv4 = UnetConv3(filters[3], filters[4], self.is_dethwise, 2 * filters[3],
                               nn.LeakyReLU(inplace=True), self.att4, self.is_res, kernel_size=(3, 3, 3),
                               padding_size=(1, 1, 1),
                               stride=(1, 1, 1))

        if is_pooling:
            self.downconv0 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.downconv1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            self.downconv2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            self.downconv3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            self.downconv4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        else:
            self.downconv0 = nn.Conv3d(in_channels=filters[0], out_channels=filters[0], kernel_size=(3, 3, 1),
                                       stride=(2, 2, 1), padding=(1, 1, 0), groups=1, dilation=(1, 1, 1))
            self.downconv1 = nn.Conv3d(in_channels=filters[1], out_channels=filters[1], kernel_size=(3, 3, 3),
                                       stride=(2, 2, 2), padding=(1, 1, 1), groups=1, dilation=(1, 1, 1))
            self.downconv2 = nn.Conv3d(in_channels=filters[2], out_channels=filters[2], kernel_size=(3, 3, 3),
                                       stride=(2, 2, 2), padding=(1, 1, 1), groups=1, dilation=(1, 1, 1))
            self.downconv3 = nn.Conv3d(in_channels=filters[3], out_channels=filters[3], kernel_size=(3, 3, 3),
                                       stride=(2, 2, 2), padding=(1, 1, 1), groups=1, dilation=(1, 1, 1))
            self.downconv4 = nn.Conv3d(in_channels=filters[4], out_channels=filters[4], kernel_size=(3, 3, 3),
                                       stride=(2, 2, 2), padding=(1, 1, 1), groups=1, dilation=(1, 1, 1))


        self.center = UnetConv3(filters[4], filters[5], self.is_dethwise, filters[4],
                               nn.LeakyReLU(inplace=True), self.att5, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                               stride=(1, 1, 1))
        self.gating = UnetGridGatingSignal3(filters[5], filters[5], kernel_size=(1, 1, 1))

         # attention blocks
        #self.attentionblock0 = MultiAttentionBlock(in_size=filters[0], gate_size=filters[1], inter_size=filters[1],
                                                   #sub_sample_factor= attention_dsample)
        self.attentionblock1 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   sub_sample_factor= attention_dsample)
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[4], gate_size=filters[5], inter_size=filters[4],
                                                   sub_sample_factor= attention_dsample)

        # upsampling

        self.up_concat4 = UnetUp3(filters[5], filters[4],  self.is_dethwise, filters[5],
                                  nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                                  stride=(1, 1, 1), scale_factor=(2, 2, 2))
        self.up_concat3 = UnetUp3(filters[4], filters[3], self.is_dethwise, filters[4],
                                  nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                                  stride=(1, 1, 1), scale_factor=(2, 2, 2))
        self.up_concat2 = UnetUp3(filters[3], filters[2], self.is_dethwise, 2*filters[3],
                                  nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                                  stride=(1, 1, 1), scale_factor=(2, 2, 2))
        self.up_concat1 = UnetUp3(filters[2], filters[1], self.is_dethwise, 2*filters[2],
                                  nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                                  stride=(1, 1, 1), scale_factor=(2, 2, 2))
        self.up_concat0 = UnetUp3(filters[1], filters[0], self.is_dethwise, 2*filters[1],
                                  nn.LeakyReLU(inplace=True), None, self.is_res, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                                  stride=(1, 1, 1),scale_factor=(2, 2, 1))

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[4], out_size=n_classes, scale_factor=[16,16,8])  #notice scale_factor
        self.dsv3 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=[8,8,4])
        self.dsv2 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=[4,4,2])
        self.dsv1 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=[2,2,1])
        self.dsv0 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*5, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv0 = self.conv0(inputs)
        downconv0 = self.downconv0(conv0)

        conv1 = self.conv1(downconv0)
        downconv1 = self.downconv1(conv1)

        conv2 = self.conv2(downconv1)
        downconv2 = self.downconv2(conv2)

        conv3 = self.conv3(downconv2)
        downconv3 = self.downconv3(conv3)

        conv4 = self.conv4(downconv3)
        downconv4 = self.downconv4(conv4)

        # Gating Signal Generation
        center = self.center(downconv4)
        gating = self.gating(center)

        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        g_conv1, att1 = self.attentionblock1(conv1, up2)
        up1 = self.up_concat1(g_conv1, up2)
        #g_conv0, att0 = self.attentionblock0(conv0, up1)
        up0 = self.up_concat0(conv0, up1)
        
        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv0 = self.dsv0(up0)
        final = self.final(torch.cat([dsv0,dsv1,dsv2,dsv3,dsv4], dim=1))

        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, 
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1


