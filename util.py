import csv
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
import json
import collections
from transforms import Transformations
import cv2
import SimpleITK as sitk


def load_csv(name):
    csvFile = open(name, 'r')
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        data.append(item)
    csvFile.close()
    return data

def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())

def write_nrrd_img(img,origin,spacing,direction):
    img_save = np.transpose(img,(2,0,1))  # hwd -- dhw
    img_save = sitk.GetImageFromArray(img_save)
    img_save.SetOrigin(origin)
    img_save.SetSpacing(spacing)
    img_save.SetDirection(direction)
    return img_save


def get_dataset_transformation(name, opts=None):
    '''
    :param opts: augmentation parameters
    :return:
    '''
    # Build the transformation object and initialise the augmentation parameters
    trans_obj = Transformations(name)
    if opts: trans_obj.initialise(opts)

    # Print the input options
    trans_obj.print()

    # Returns a dictionary of transformations
    return trans_obj.pros_3d_transform()


def _fast_hist(label_true, label_pred, n_class):
    # mask为掩膜（去除了255这些点（即标签图中的白色的轮廓），其中的>=0是为了防止bincount()函数出错）
    mask = (label_true >= 0) & (label_true < n_class)
    # bincount()函数用于统计数组内每个非负整数的个数
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class) #0,3,6 -> 0,1,2,3,4,5,6,7,8(9种可能性)
    return hist #[3,3]


def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, fwavacc, mean_iu

def counter_distance(l_gt_cls):
    X, Y, Z = l_gt_cls.shape
    ## class_id 部分保留1
    #l_gt_cls = np.array(l_gt == class_id, dtype=np.float32)
    final_dis = []
    # 循环每层切片
    for z in range(Z):
        dis_list = []
        # 如果有轮廓
        slice_gt_cls = l_gt_cls[:,:,z]
        if np.sum(slice_gt_cls) > 0:
            contours, _ = cv2.findContours(cv2.inRange(slice_gt_cls, 1, 1),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
            for i in range(X):
                #保存每行
                tmp_line = []
                for j in range(Y):
                    dist = cv2.pointPolygonTest(contours[0], (i, j), True)
                    dist = np.abs(dist)
                    tmp_line.append(dist)
                #合并每行为切片
                dis_list.append(tmp_line)
        else:
            #dis_list = np.zeros_like((X,Y)).astype(np.float)
            #dis_list = [[0] * Y ] * X
            dis_list = np.zeros([X,Y],dtype=float)

        #合并每层切片
        final_dis.append(dis_list)
    final_dis = np.array(final_dis)
    final_dis = np.transpose(final_dis,(1,2,0))
    min = np.amin(final_dis)
    max = np.amax(final_dis)
    wdis_map = (final_dis - min) / (max - min + 0.000001)
    return 1 + 9*wdis_map


def weight_dice_score_list(label_gt, label_pred, n_class):
    """
    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    w_dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    w_PPV = np.zeros((batchSize, n_class), dtype=np.float32) #Positive predictive value,precision
    w_SEN = np.zeros((batchSize, n_class), dtype=np.float32) #Sensitivity,recall
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32)
            if class_id == 0:
                edge_dis = 1
            else:
                edge_dis = counter_distance(img_A).flatten()
            img_A = img_A.flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis*img_A) + np.sum(edge_dis*img_B) + epsilon)
            ppv_score = np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis * img_B) + epsilon)
            sen_score = np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis * img_A) + epsilon)
            w_dice_scores[batch_id, class_id] = score
            w_PPV[batch_id, class_id] = ppv_score
            w_SEN[batch_id, class_id] = sen_score

    return np.mean(w_dice_scores, axis=0),np.mean(w_PPV, axis=0),np.mean(w_SEN, axis=0)

def dice_score_list(label_gt, label_pred, n_class):
    """
    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    PPV = np.zeros((batchSize, n_class), dtype=np.float32) #Positive predictive value,precision
    SEN = np.zeros((batchSize, n_class), dtype=np.float32) #Sensitivity,recall
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            ppv_score = np.sum(img_A * img_B) / (np.sum(img_B) + epsilon)
            sen_score = np.sum(img_A * img_B) / (np.sum(img_A) + epsilon)
            dice_scores[batch_id, class_id] = score
            PPV[batch_id, class_id] = ppv_score
            SEN[batch_id, class_id] = sen_score

    return np.mean(dice_scores, axis=0),np.mean(PPV, axis=0),np.mean(SEN, axis=0)


def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.size(1)
    pred_lbls = pred_seg.data.max(1)[1].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)  #[2,128,128,48]
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_) #[[128,128,48],[128,128,48]]
        preds.append(pred_)

    Overall_Acc, mean_acc, fwavacc, mean_iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice,ppv,sen = dice_score_list(gts, preds, n_class=n_classes)
    #w_dice, w_ppv, w_sen = weight_dice_score_list(gts, preds, n_class=n_classes)
    #precision, recall = precision_and_recall(gts, preds, n_class=n_classes)

    return Overall_Acc, mean_acc, mean_iou, dice,ppv,sen

def weight_dice_score_def(label_gt, label_pred, n_class):
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    w_dice_scores = np.zeros(n_class, dtype=np.float32)
    w_PPV = np.zeros(n_class, dtype=np.float32)  # Positive predictive value
    w_SEN = np.zeros(n_class, dtype=np.float32)  # Sensitivity
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32)
        if class_id == 0:
            edge_dis = 1
        else:
            edge_dis = counter_distance(img_A).flatten()
        img_A = img_A.flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis * img_A) + np.sum(edge_dis * img_B) + epsilon)
        ppv_score = np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis * img_B) + epsilon)
        sen_score = np.sum(edge_dis * img_A * img_B) / (np.sum(edge_dis * img_A) + epsilon)
        w_dice_scores[class_id] = score
        w_PPV[class_id] = ppv_score
        w_SEN[class_id] = sen_score

    return w_dice_scores,w_PPV,w_SEN

def dice_score_def(label_gt, label_pred, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    PPV = np.zeros(n_class, dtype=np.float32)  # Positive predictive value
    SEN = np.zeros(n_class, dtype=np.float32)  # Sensitivity
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        ppv_score = np.sum(img_A * img_B) / (np.sum(img_B) + epsilon)
        sen_score = np.sum(img_A * img_B) / (np.sum(img_A) + epsilon)
        dice_scores[class_id] = score
        PPV[class_id] = ppv_score
        SEN[class_id] = sen_score

    return dice_scores,PPV,SEN


def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall


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


def distance_metric(seg_A, seg_B, dx, k):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """

    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k) #gt
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []
    table_asd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            image, contours, hierarchy = cv2.findContours(cv2.inRange(slice_A, 1, 1), cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_NONE)

            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            image, contours, hierarchy = cv2.findContours(cv2.inRange(slice_B, 1, 1), cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_NONE)

            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance

            temp1 = np.min(M, axis=0)
            temp2 = np.min(M, axis=1)
            md = 1/len(temp1) * (np.sum(temp1)) * dx
            asd = 1/(len(temp1)+len(temp2)) * (np.sum(temp1) + np.sum(temp2)) * dx
            hd = np.max([select(temp1, int(len(temp1) * 0.95)), select(temp2, int(len(temp2) * 0.95))]) * dx  # 95HD
            # hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_asd += [asd]
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_asd = np.mean(table_asd) if table_asd else 0
    mean_md = np.mean(table_md) if table_md else 0
    mean_hd = np.mean(table_hd) if table_hd else 0
    return  mean_md, mean_hd, mean_asd


def get_scheduler(optimizer, opt):
    # print('opt.lr_policy = [{}]'.format(opt.lr_policy))
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        # print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, threshold=0.01, patience=20)
    elif opt.lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



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
        # print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        # print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = nn.functional.interpolate(input=(self.outputs[index]).data(),size=newsize[2:], mode='bilinear',align_corners=True)
        else:
            self.outputs = nn.functional.interpolate(input=(self.outputs).data(),size=newsize[2:], mode='bilinear',align_corners=True)

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)

        detach1 = Variable(torch.IntTensor(0))
        detach2 = Variable(torch.IntTensor(0))
        self.submodule(x, detach1, detach2)

        # self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class HookBasedSingleFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedSingleFeatureExtractor, self).__init__()

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
        # print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        # print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.functional.interpolate(size=newsize[2:], mode='bilinear',align_corners=True)
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = nn.functional.interpolate(input=(self.outputs[index]).data(),size=newsize[2:], mode='bilinear',align_corners=True)
        else:
            self.outputs = nn.functional.interpolate(input=(self.outputs).data(),size=newsize[2:], mode='bilinear',align_corners=True)

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)

        self.submodule(x)

        # self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


