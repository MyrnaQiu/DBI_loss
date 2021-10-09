#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:51:29 2018

@author: myrna
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import math
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import os
from scipy.special import softmax


class Dice_cross_entropy_3D(nn.Module):
    def __init__(self, n_classes, alpha=None, size_average=True):
        super(Dice_cross_entropy_3D, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([0.2, alpha, 0.8 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input1 = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target1 = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        ## dice score
        inter = torch.sum(input1 * target1, 2) + smooth
        # print(inter.shape)   [batch, deth]
        union = torch.sum(input1, 2) + torch.sum(target1, 2) + smooth
        # print(union.shape) [batch, deth]
        score = torch.sum(2.0 * inter / union)
        dice_score = score / (float(batch_size) * float(self.n_classes))

        ## focal loss
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * Variable(at)

        ce_loss = -1 * logpt
        if self.size_average:
            ce_score = ce_loss.mean()
        else:
            ce_score = ce_loss.sum()

        loss = ce_score + (1.0 - dice_score)
        return loss

class DiceFocalLoss(nn.Module):
    def __init__(self, n_classes,weight=0.3, gamma=2,alpha=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.weight = torch.cuda.FloatTensor([weight,1-weight])
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([0.2,alpha,0.8-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        
    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input1 = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target1 = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        
        ## dice score
        inter = torch.sum(input1 * target1, 2) + smooth
        #print(inter.shape)   [batch, deth]
        union = torch.sum(input1, 2) + torch.sum(target1, 2) + smooth
        #print(union.shape) [batch, deth]
        score = torch.sum(2.0 * inter / union)
        dice_score = score / (float(batch_size) * float(self.n_classes))
        
        ## focal loss
        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        focal_loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            focal_score = focal_loss.mean()
        else:
            focal_score = focal_loss.sum()
            
        loss = self.weight[0] * focal_score+self.weight[0]*(1.0 - dice_score)
        return loss
    
class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([0.2,alpha,0.8-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class cross_entropy_3D(nn.Module):
    def __init__(self, alpha=None, size_average=True):
        super(cross_entropy_3D, self).__init__()
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([0.2,alpha,0.8-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1) #N*H*W

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda() #N*H*W,1
            at = self.alpha.gather(0,select.data.view(-1)) #N*H*W
            logpt = logpt * Variable(at)

        loss = -1 * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class DiceoCELoss(nn.Module):
    def __init__(self, n_classes,alpha=None, size_average=True):
        super(DiceoCELoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([0.2,alpha,0.8-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        
    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input1 = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target1 = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        
        ## dice score
        inter = torch.sum(input1 * target1, 2) + smooth
        #print(inter.shape)   [batch, deth]
        union = torch.sum(input1, 2) + torch.sum(target1, 2) + smooth
        #print(union.shape) [batch, deth]
        score = torch.sum(2.0 * inter / union)
        dice_score = score / (float(batch_size) * float(self.n_classes))
        
        ## focal loss
        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        CE_loss = -1  * logpt
        if self.size_average: 
            CE_score = CE_loss.mean()
        else:
            CE_score = CE_loss.sum()
            
        loss = CE_score + (1.0 - dice_score)
        return loss
        
class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        #input [1, 2, 160, 160, 64]
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        #print(target.shape)  #[batch, deth, x*y*z]
        #print(input.shape)   #[batch, deth, x*y*z]        

        inter = torch.sum(input * target, 2) + smooth
        #print(inter.shape)   [batch, deth]
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        #print(union.shape) [batch, deth]

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))
        #print(score)  # Variable [torch.cuda.FloatTensor of size 1 (GPU 0)]

        return score
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        #input_flat = input.contiguous().view(N, -1)
        #target_flat = target.contiguous().view(N, -1)
        intersection = input * target
        loss = 2 * (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()
        #self.ones = torch.sparse.torch.eye(depth).cuda() #[1,0;0,1]
        #self.ones = torch.FloatTensor([[0.8,0],[0,1.2]]).cuda()
        

    def forward(self, X_in):
        n_dim = X_in.dim()
        #print(n_dim) #5
        output_size = X_in.size() + torch.Size([self.depth])
        #print(output_size) #[batchsize, 1, 160, 160, 48, 2]
        num_element = X_in.numel()  #batchsize*x*y*z
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size) #view()~reshape  
        #before view [3276800, 2]
        #print(X_in.shape) #2457600
        #print(out.shape)  #[batchsize, 1, x, y, z, deth]
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float() #[batch,deth,x,y,z] [deth=0--value=1;deth=1---0]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def partition(seq):
    pi, seq = seq[0], seq[1:]  # 选取并移除主元
    lo = [x for x in seq if x <= pi]
    hi = [x for x in seq if x > pi]
    return lo, pi, hi

def select(seq, k):
    lo, pi, hi = partition(seq)
    m = len(lo)
    if m == k: return pi
    if m < k: return select(hi, k - m - 1)
    return select(lo, k)

def weight_loss(distence,pred,label):
    '''

    :param distence:
    :param pred: [0.1,0.2,0.7]
    :param label: 0,1,2
    :return: one point ce loss
    '''

    weight = distence

    weight = abs(weight)
    pred_onehot = pred[label]+0.1e-10

    w_loss = -1 * math.log(pred_onehot) * math.log10(weight + 1)

    return w_loss
    #return pow(pred - label,2) * weight

def weight_distance_metric(seg_soft,seg_P, seg_G):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """
    X, Y, Z = seg_G.shape
    weight_loss_sum = 0
    error_coordi_h = 0
    error_coordi_w = 0
    for z in range(Z):
        # Binary mask at this slice
        #print(type(seg_soft), seg_soft.shape)
        slice_soft = np.array(seg_soft)[:, :, :, z]
        slice_P = seg_P[:, :, z].astype(np.uint8)
        slice_G = seg_G[:, :, z].astype(np.uint8)

        dis_slice_sum_list = []

        if np.sum(slice_G) > 0:

            if np.sum(slice_G[slice_G==1]) > 0:
                slice_P1 = slice_P.copy()
                slice_P1[slice_P1 == 2] = 0
                slice_G1 = slice_G.copy()
                slice_G1[slice_G1 == 2] = 0
                contours1, _ = cv2.findContours(cv2.inRange(slice_G1, 1, 1),
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_NONE)
                slice_error1 = slice_G1 - slice_P1
                pts_error1 = np.where(slice_error1 != 0)
                error_coordi_h = pts_error1[0]
                error_coordi_w = pts_error1[1]
                for i in range(len(pts_error1[0])):
                    dist1 = cv2.pointPolygonTest(contours1[0], (pts_error1[1][i], pts_error1[0][i]), True)
                    dist1 = np.abs(dist1)
                    dis_slice_sum_list.append(dist1)
                    # print(dist1)
            if np.sum(slice_G[slice_G == 2]) > 0:
                slice_P2 = slice_P.copy()
                slice_P2[slice_P2 == 1] = 0
                slice_G2 = slice_G.copy()
                slice_G2[slice_G2 == 1] = 0
                contours2, _ = cv2.findContours(cv2.inRange(slice_G2, 2, 2),
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_NONE)
                slice_error2 = slice_G2 - slice_P2
                pts_error2 = np.where(slice_error2 != 0)
                error_coordi_h = np.hstack((error_coordi_h, pts_error2[0]))
                error_coordi_w = np.hstack((error_coordi_w, pts_error2[1]))
                for i in range(len(pts_error2[0])):
                    dist2 = cv2.pointPolygonTest(contours2[0], (pts_error2[1][i], pts_error2[0][i]), True)
                    dist2 = np.abs(dist2)
                    dis_slice_sum_list.append(dist2)

            # print('coor:',error_coordi_h,error_coordi_w)
            # print('dis',dis_slice_sum_list)

            for i in range(len(dis_slice_sum_list)):
                h = error_coordi_h[i]
                w = error_coordi_w[i]
                weight_loss_sum += weight_loss(dis_slice_sum_list[i], slice_soft[:,h,w], slice_G[h,w])

    return weight_loss_sum


def tensor_weight_distance_metric_norm(seg_P, seg_G, nclass):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        seg_G,seg_P: H*W*D
        """
    X, Y, Z = seg_G.shape
    weight_map = np.zeros_like(seg_G).astype(np.float)

    # 循环每层切片
    for z in range(Z):
        # Binary mask at this slice
        #print(type(seg_soft), seg_soft.shape)
        error_coordi_h = []
        error_coordi_w = []
        dis_list = []
        slice_P = seg_P[:, :, z].astype(np.uint8)
        slice_G = seg_G[:, :, z].astype(np.uint8)
        #如果有轮廓
        if np.sum(slice_G) > 0:
            #轮廓1

            for c in range(nclass):
                if np.sum(slice_G[slice_G == c]) > 0:

                    slice_P1= np.array(slice_P == c, dtype=np.float32)
                    slice_G1= np.array(slice_G == c, dtype=np.float32)
                    #获取轮廓1
                    _, contours1, _ = cv2.findContours(cv2.inRange(slice_G1, 1, 1),
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
                    #相减，非零位置为错误
                    slice_error1 = slice_G1 - slice_P1
                    #获取error位置，error_coordi_h，error_coordi_w
                    pts_error1 = np.where(slice_error1 != 0)
                    error_coordi_h.append(pts_error1[0])
                    error_coordi_w.append(pts_error1[1])
                    #获取error距离
                    for i in range(len(pts_error1[0])):
                        dist1 = cv2.pointPolygonTest(contours1[0], (pts_error1[1][i], pts_error1[0][i]), True)
                        dist1 = np.abs(dist1)
                        dis_list.append(dist1)
                        #weight_map[pts_error1[0],pts_error1[1],z] += dist1

                    # print(dist1)
        elif np.sum(slice_G) == 0 and np.sum(slice_P)>0:
            pts_error3 = np.where(slice_P != 0)
            error_coordi_h.append(pts_error3[0])
            error_coordi_w.append(pts_error3[1])
            for i in range(len(pts_error3[0])):
                dis_list.append(10)

        for h,w,dist in zip(error_coordi_h,error_coordi_w,dis_list):
            weight_map[h, w, z] += dist
        min = np.amin(weight_map[:,:,z])
        max = np.amax(weight_map[:,:,z])
        weight_map[:,:,z] = (weight_map[:,:,z] - min) / (max - min + 0.000001)
        #weight_map2 = 2*(np.exp(weight_map)-1)
        weight_map2 = np.exp(weight_map) - 1

    return weight_map2


class dbi_cross_entropy_3D(nn.Module):
    def __init__(self, n_classes,size_average=False):
        super(dbi_cross_entropy_3D, self).__init__()
        self.size_average = size_average
        self.n_classes = n_classes

    def forward(self, input, target):
        if input.dim() > 2:
            input_ce = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input_ce = input_ce.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_ce = input_ce.contiguous().view(-1, input_ce.size(2))  # N,H*W,C => N*H*W,C
        target_ce = target.view(-1, 1)

        logpt = F.log_softmax(input_ce,dim=1)
        logpt = logpt.gather(1, target_ce)
        logpt = logpt.view(-1).float()  # N*H*W

        #weight_distance
        batch_size = input.size(0)
        logits = F.softmax(input, dim=1)
        weight_map = []

        for i in range(batch_size):
            pred_seg = logits[i].data.max(0)[1].cpu().numpy() #h,w,d
            #for j in range(len(self.n_classes)):
            #a = target[i].shape
            wdis_map = tensor_weight_distance_metric_norm(pred_seg, target[i][0].cpu().numpy(),self.n_classes) #each slice dis sum
            #min = np.amin(wdis_map)
            #max = np.amax(wdis_map)
            #wdis_map = (wdis_map - min) / (max - min + 0.000001)
            if i == 0:
                weight_map = wdis_map
                '''
                wmap_visual = np.zeros_like(weight_map[:,:,0])
                if is_visual:
                    for d in range(48):
                        weight_map1 = weight_map[:,:,d]
                        weight_map1 = weight_map1**2
                        wmap_visual += weight_map1
                    heatmap1 = cv2.applyColorMap((np.sqrt(wmap_visual)* 255).astype(np.uint8), cv2.COLORMAP_JET)
                    save_path ='visual/edge/{}'.format(name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,'epoch_{}.jpg'.format(epoch)),heatmap1)
                '''
            else:
                weight_map = np.array([weight_map,wdis_map]) #batch,H,W,D

        #if weight_map.type() != input.data.type():
            #weight_map = weight_map.type_as(input.data)
        weight_map = torch.from_numpy(weight_map).float().cuda()
        #weight_map = F.softmax(weight_map, dim=0)
        weight_map = weight_map +1
        weight_map = weight_map.view(-1)

        logpt = logpt * weight_map

        loss = -1 * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class dbi_focal_3D(nn.Module):
    def __init__(self, n_classes,gamma=2, size_average=False):
        super(dbi_focal_3D, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.n_classes = n_classes
    def forward(self, input, target):
        if input.dim() > 2:
            input_ce = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input_ce = input_ce.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_ce = input_ce.contiguous().view(-1, input_ce.size(2))  # N,H*W,C => N*H*W,C
        target_ce = target.view(-1, 1)

        logpt = F.log_softmax(input_ce,dim=1)
        logpt = logpt.gather(1, target_ce)
        logpt = logpt.view(-1).float()  # N*H*W
        pt = logpt.data.exp()

        #weight_distance
        batch_size = input.size(0)
        logits = F.softmax(input, dim=1)
        weight_map = []
        for i in range(batch_size):
            pred_seg = logits[i].data.max(0)[1].cpu().numpy() #h,w,d
            #for j in range(len(self.n_classes)):
            #a = target[i].shape
            wdis_map = tensor_weight_distance_metric_norm(pred_seg, target[i][0].cpu().numpy(),self.n_classes) #each slice dis sum
            #min = np.amin(wdis_map)
            #max = np.amax(wdis_map)
            #wdis_map = (wdis_map - min) / (max - min + 0.000001)
            if i == 0:
                weight_map = wdis_map
            else:
                weight_map = np.array([weight_map,wdis_map]) #batch,H,W,D

        #if weight_map.type() != input.data.type():
            #weight_map = weight_map.type_as(input.data)
        weight_map = torch.from_numpy(weight_map).float().cuda()
        #weight_map = F.softmax(weight_map, dim=0)
        weight_map = weight_map +1
        weight_map = weight_map.view(-1)

        logpt = logpt * weight_map

        loss = -1 * (1-pt)**self.gamma*logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Softdice_dbi_cross_entropy_3D(nn.Module):
    def __init__(self, n_classes,e_weight,size_average=False):
        super(Softdice_dbi_cross_entropy_3D, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.e_weight = e_weight
        self.size_average = size_average

    def forward(self, input, target):
        #dice loss
        smooth = 0.01
        batch_size = input.size(0)
        # input [1, 2, 160, 160, 64]
        input_dice = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target_dice = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input_dice * target_dice, 2) + smooth
        # print(inter.shape)   [batch, deth]
        union = torch.sum(input_dice, 2) + torch.sum(target_dice, 2) + smooth
        # print(union.shape) [batch, deth]
        score_dice = torch.sum(2.0 * inter / union)

        #ce loss
        if input.dim() > 2:
            input_ce = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input_ce = input_ce.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_ce = input_ce.contiguous().view(-1, input_ce.size(2))  # N,H*W,C => N*H*W,C
        target_ce = target.view(-1, 1)

        logpt = F.log_softmax(input_ce,dim=1)
        logpt = logpt.gather(1, target_ce)
        logpt = logpt.view(-1).float()  # N*H*W

        #weight_distance
        batch_size = input.size(0)
        logits = F.softmax(input, dim=1)
        weight_map = []
        for i in range(batch_size):
            pred_seg = logits[i].data.max(0)[1].cpu().numpy() #h,w,d
            #for j in range(len(self.n_classes)):
            #a = target[i].shape
            wdis_map = tensor_weight_distance_metric_norm(pred_seg, target[i][0].cpu().numpy(),self.n_classes) #each slice dis sum
            #min = np.amin(wdis_map)
            #max = np.amax(wdis_map)
            #wdis_map = (wdis_map - min) / (max - min + 0.000001)
            if i == 0:
                weight_map = wdis_map
            else:
                weight_map = np.array([weight_map,wdis_map]) #batch,H,W,D

        #if weight_map.type() != input.data.type():
            #weight_map = weight_map.type_as(input.data)
        weight_map = torch.from_numpy(weight_map).float().cuda()
        #weight_map = F.softmax(weight_map, dim=0)
        weight_map = weight_map*self.e_weight +1
        weight_map = weight_map.view(-1)

        logpt = logpt * weight_map

        loss_ce = -1 * logpt
        if self.size_average:
            score_ce = loss_ce.mean()
        else:
            score_ce = loss_ce.sum()

        score = 1.0 - score_dice / (float(batch_size) * float(self.n_classes)) + score_ce
        return score

class Softdice_dbiFocal_loss(nn.Module):
    def __init__(self, n_classes,e_weight,gamma=2,size_average=False):
        super(Softdice_dbiFocal_loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.e_weight = e_weight
        self.size_average = size_average
        self.gamma = gamma

    def forward(self, input, target):
        #dice loss
        smooth = 0.01
        batch_size = input.size(0)
        # input [1, 2, 160, 160, 64]
        input_dice = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target_dice = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input_dice * target_dice, 2) + smooth
        # print(inter.shape)   [batch, deth]
        union = torch.sum(input_dice, 2) + torch.sum(target_dice, 2) + smooth
        # print(union.shape) [batch, deth]
        score_dice = torch.sum(2.0 * inter / union)

        #ce loss
        if input.dim() > 2:
            input_ce = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input_ce = input_ce.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_ce = input_ce.contiguous().view(-1, input_ce.size(2))  # N,H*W,C => N*H*W,C
        target_ce = target.view(-1, 1)

        logpt = F.log_softmax(input_ce,dim=1)
        logpt = logpt.gather(1, target_ce)
        logpt = logpt.view(-1).float()  # N*H*W
        pt = logpt.data.exp()

        #weight_distance
        batch_size = input.size(0)
        logits = F.softmax(input, dim=1)
        weight_map = []
        for i in range(batch_size):
            pred_seg = logits[i].data.max(0)[1].cpu().numpy() #h,w,d
            #for j in range(len(self.n_classes)):
            #a = target[i].shape
            wdis_map = tensor_weight_distance_metric_norm(pred_seg, target[i][0].cpu().numpy(),self.n_classes) #each slice dis sum
            #min = np.amin(wdis_map)
            #max = np.amax(wdis_map)
            #wdis_map = (wdis_map - min) / (max - min + 0.000001)
            if i == 0:
                weight_map = wdis_map
            else:
                weight_map = np.array([weight_map,wdis_map]) #batch,H,W,D

        #if weight_map.type() != input.data.type():
            #weight_map = weight_map.type_as(input.data)
        weight_map = torch.from_numpy(weight_map).float().cuda()
        #weight_map = F.softmax(weight_map, dim=0)
        weight_map = weight_map*self.e_weight +1
        weight_map = weight_map.view(-1)

        logpt = logpt * weight_map

        loss_focal = -1 * (1-pt)**self.gamma*logpt
        if self.size_average:
            score_focal = loss_focal.mean()
        else:
            score_focal = loss_focal.sum()

        score = 1.0 - score_dice / (float(batch_size) * float(self.n_classes)) + score_focal
        return score
