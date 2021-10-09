#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:58:08 2019

@author: myrna
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from adabound import AdaBound
import torch.nn.functional as F
import os
from os.path import join
from os import listdir
import numpy as np
import torchsample.transforms as ts
from Pros3DDataset_roi import Pros3DDataset
from tqdm import tqdm

from model.unet3d_dsv_ag_att import unet3d_dsv_ag_att
from model.unet3d_dsv_noagatt import unet3d_dsv_noagatt

from model.networks_other import print_model

from loss import *

from util import *
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
import random
from boundary_loss import *
import SimpleITK as sitk


class main_seg():
    def __init__(self, json_filename):
        super(main_seg, self).__init__()
        self.json_filename = json_filename
        # Load options
        self.json_opts = json_file_to_pyobj(self.json_filename)
        self.train_opts = self.json_opts.training
        self.csv_name = self.train_opts.csv_name
        self.model_opts = self.json_opts.model
        self.model_type = self.json_opts.model.model_type
        self.model_name = self.json_opts.model.model_name
        self.model_pretype = self.json_opts.model.model_pretype
        self.ds_path = self.json_opts.path.data_path
        self.crop_shape = self.json_opts.augmentation.pros.scale_size
        self.val_scale_size = self.json_opts.augmentation.pros.val_scale_size
        self.patience = self.json_opts.model.patience
        self.visual_epoch = self.train_opts.visual_epoch
        self.sum_type = self.json_opts.model.sum_type
        self.e_weight = self.json_opts.model.e_weight
        self.gamma = self.json_opts.model.gamma
        self.alpha = self.json_opts.model.alpha
        self.setup_seed(20)

        self.device_ids = self.model_opts.gpu_ids
        os.environ['CUDA_VISIBLE_DEVICES'] = self.device_ids

        if self.model_name == 'unet3d_dsv_ag_att':
            self.model = unet3d_dsv_ag_att(feature_scale=self.model_opts.feature_scale,
                                           n_classes=self.model_opts.n_classes,
                                           in_channels=self.model_opts.input_nc, attention_dsample=(2, 2, 2),
                                           is_pooling=self.model_opts.is_pooling,
                                           is_dethwise=self.model_opts.is_dethwise,
                                           attmodule=self.model_opts.attmodule,
                                           is_res=self.model_opts.is_res).cuda()
        elif self.model_name == 'unet3d_dsv_noag_att':
            self.model = unet3d_dsv_noagatt(feature_scale=self.model_opts.feature_scale,
                                            n_classes=self.model_opts.n_classes,
                                            in_channels=self.model_opts.input_nc, attention_dsample=(2, 2, 2),
                                            is_pooling=self.model_opts.is_pooling,
                                            is_dethwise=self.model_opts.is_dethwise,
                                            attmodule=self.model_opts.attmodule,
                                            is_res=self.model_opts.is_res).cuda()

        if self.train_opts.is_train == True:
            self.optimizer = []

            if self.json_opts.model.criterion == 'dice_focal':
                self.criterion = DiceFocalLoss(n_classes=self.model_opts.output_nc, weight=0.5, gamma=self.gamma, alpha=self.alpha, size_average=True)
            elif self.json_opts.model.criterion == 'dice':
                self.criterion = SoftDiceLoss(self.model_opts.output_nc)
            elif self.json_opts.model.criterion == 'focal':
                self.criterion = FocalLoss(gamma=self.gamma, alpha=self.alpha, size_average=True)
            elif self.json_opts.model.criterion == 'ce':
                self.criterion = cross_entropy_3D(alpha=None, size_average=True)
            elif self.json_opts.model.criterion == 'dice_ce':
                self.criterion = Dice_cross_entropy_3D(self.model_opts.output_nc, alpha=None, size_average=True)
            elif self.json_opts.model.criterion == 'dbi_ce':
                self.criterion = dbi_cross_entropy_3D(self.model_opts.output_nc, size_average=True)
            elif self.json_opts.model.criterion == 'dbi_focal':
                self.criterion = dbi_focal_3D(self.model_opts.output_nc, gamma=self.gamma, size_average=True)
            elif self.json_opts.model.criterion == 'dice_dbi_ce':
                self.criterion = Softdice_dbi_cross_entropy_3D(self.model_opts.output_nc, self.e_weight,
                                                               size_average=True)
            elif self.json_opts.model.criterion == 'dice_dbi_focal':
                self.criterion = Softdice_dbiFocal_loss(self.model_opts.output_nc, self.e_weight, gamma=self.gamma,
                                                        size_average=True)

            elif self.json_opts.model.criterion == 'boundary':
                self.criterion = SurfaceLoss()
                self.criterion2 = SoftDiceLoss(self.model_opts.output_nc)
            elif self.json_opts.model.criterion == 'dice_HD':
                self.criterion = HausdorffLoss()
                self.criterion2 = SoftDiceLoss(self.model_opts.output_nc)

            self.optimizer = self.get_optimizer(self.model_opts.optimizer_opt,
                                                filter(lambda p: p.requires_grad, self.model.parameters()))

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_optimizer(self, option, params):
        # opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
        if option == 'sgd':
            opt = optim.SGD(params,
                            lr=self.model_opts.lr_rate,
                            momentum=0.9,
                            nesterov=True,
                            weight_decay=self.model_opts.l2_reg_weight)

        if option == 'adam':
            opt = optim.Adam(params,
                             lr=self.model_opts.lr_rate,
                             betas=(0.9, 0.999),
                             weight_decay=self.model_opts.l2_reg_weight)
        if option == 'adabound':
            opt = AdaBound(params, lr=self.model_opts.lr_rate, final_lr=0.1)

        return opt

    '''
    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.scheduler.append(get_scheduler(self.optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(self.optimizer))
    '''

    def train(self):

        # Architecture type
        arch_type = self.train_opts.arch_type  # "arch_type": "pros":

        model_file = '{}.pkl'.format(self.model_type)
        model_file_best = '{}_best.pkl'.format(self.model_type)
        model_pretrain = '{}.pkl'.format(self.model_pretype)

        save_path = os.path.join(os.getcwd(), 'save/seg/boundary', self.model_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        '''
        if not os.path.exists(model_file):
            os.makedirs(model_file)
        if not os.path.exists(model_file_best):
            os.makedirs(model_file_best)
        '''

        # Setup Dataset and Augmentation
        ds_transform = get_dataset_transformation(arch_type, opts=self.json_opts.augmentation)

        # Setup Data Loader
        train_dataset = Pros3DDataset(self.ds_path, split='train', crop_shape=self.crop_shape, csv_name=self.csv_name,
                                      transform=ds_transform['train'])
        test_dataset = Pros3DDataset(self.ds_path, split='test', crop_shape=self.val_scale_size, csv_name=self.csv_name,
                                     transform=ds_transform['valid'])

        train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=self.train_opts.batchSize,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)

        num_parms = print_model(self.model)

        if self.train_opts.is_train == True:

            bestmodel = 10000000
            best_epoch = 0
            total_epoch = self.train_opts.n_epochs
            writer = SummaryWriter('runs/seg/{}'.format(self.model_type))
            stop_num = 0
            # self.set_scheduler(self.train_opts)

            early_stop = False
            is_visual = False
            is_visualedge = False

            if self.train_opts.is_pretrain == True:
                self.model.load_state_dict((torch.load(model_pretrain)))

            # start_time = time.perf_counter()
            times = 0
            for epoch in range(total_epoch):

                epoch_start = time.perf_counter()
                if not early_stop:

                    epoch_loss = 0
                    epoch_dice1, epoch_dice2 = 0, 0
                    epoch_Overall_Acc = 0
                    epoch_mean_acc = 0
                    epoch_mean_iou = 0
                    a = 1

                    '''
                    if epoch==0 or (epoch + self.visual_epoch + 1) % self.visual_epoch == 0:
                        is_visual = False
                        is_visualedge = True
                    else:
                        is_visual = False
                        is_visualedge = False
                    '''

                    # Number = len(train_loader)
                    # Training Iterations
                    for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                        # for epoch_iter, (images, labels) in enumerate(train_loader, 1):
                        # iter_start = time.perf_counter()
                        images, labels = images.cuda(), labels.cuda()
                        # print(labels.shape)#[batchsize, 1, 160, 160, 48]
                        # Forward + Backward + Optimize

                        self.optimizer.zero_grad()
                        outputs = self.model(images)  # print(outputs.shape)   #[batchsize, 2, 160, 160, 48]

                        #loss = self.criterion(outputs, labels)
                        seg = labels[:, 0, :, :, :]  # bwHd
                        seg2 = class2one_hot(seg, 3)  # bcwHd

                        logits = F.softmax(outputs, dim=1)  # bcwHd
                        with torch.no_grad():
                            seg5 = torch.zeros_like(logits, dtype=torch.float32)
                            for i in range(2):
                                seg3 = seg2[i].cpu().numpy()  # chwd
                                seg4 = one_hot2dist(seg3)
                                seg4 = torch.tensor(seg4)
                                seg5[i,:,:,:,:] = seg4

                        loss = (1-a)*self.criterion(logits, seg5.cuda()) + a*self.criterion2(outputs, labels)
                        epoch_loss += loss.item()
                        loss.backward()
                        self.optimizer.step()

                        Overall_Acc, mean_acc, mean_iou, dice_score, ppv, sen = segmentation_stats(outputs, labels)
                        epoch_dice1 += dice_score[1]
                        epoch_dice2 += dice_score[2]
                        epoch_Overall_Acc += Overall_Acc
                        epoch_mean_acc += mean_acc
                        epoch_mean_iou += mean_iou

                    a -= 0.003

                        # if is_visual == True and (epoch_iter == 2 or epoch_iter == 35):
                        # layer_name = 'attentionblock1'
                        # saveSingleHeatmapTrain(self.model, layer_name, images, epoch=epoch, iter=epoch_iter,
                        # filename=self.model_type)

                    epoch_end = time.perf_counter()
                    times += epoch_end-epoch_start
                    print("Epoch [%d/%d], train_Loss: %.4f" % (epoch + 1, total_epoch, epoch_loss / epoch_iter))
                    print("dice_score1:%.4f " % (epoch_dice1 / epoch_iter))
                    print("dice_score2:%.4f " % (epoch_dice2 / epoch_iter))
                    ##save parameters
                    torch.save(self.model.state_dict(), model_file)

                    # print('the epoch takes time:',epoch_end-epoch_start)

                    if epoch_loss / epoch_iter <= bestmodel:
                        torch.save(self.model.state_dict(), model_file_best)
                        bestmodel = epoch_loss / epoch_iter
                        best_epoch = epoch + 1
                        stop_num = 0
                    else:
                        stop_num += 1

                    print('best epoch:', best_epoch)
                    print('current best loss:', bestmodel)
                    print('stop_num:', stop_num)

                    if stop_num >= self.patience:
                        early_stop = True

                    '''
                    # Update the model learning rate
                    for sc in self.scheduler:       
                        sc.step(epoch_loss)
                    lr = self.optimizers[0].param_groups[0]['lr']         
                    print('current learning rate = %.7f' % lr)
                    '''

                    ## test
                    test_dice1, test_dice2, test_acc, test_mean_iou = self.test(self.model, test_loader, model_file,
                                                                                epoch=epoch + 1, visual_att=is_visual,
                                                                                visual_edge=is_visualedge,
                                                                                is_attsave=False, is_save=False)
                    print("test_dice_score1:%.4f " % (test_dice1))
                    print("test_dice_score2:%.4f " % (test_dice2))

                    ##visual
                    writer.add_scalar('data/train_loss', epoch_loss / epoch_iter, epoch)
                    writer.add_scalar('data/train_Dice Coefficient1', epoch_dice1 / epoch_iter, epoch)
                    writer.add_scalar('data/train_Dice Coefficient2', epoch_dice2 / epoch_iter, epoch)
                    writer.add_scalar('data/Overall Accuracy', epoch_Overall_Acc / epoch_iter, epoch)
                    writer.add_scalar('data/Mean Accuracy', epoch_mean_acc / epoch_iter, epoch)
                    writer.add_scalar('data/Mean IoU', epoch_mean_iou / epoch_iter, epoch)

                    writer.add_scalar('data/test_Dice Coefficient1', test_dice1, epoch)
                    writer.add_scalar('data/test_Dice Coefficient2', test_dice2, epoch)
                    writer.add_scalar('data/test_Mean Accuracy', test_acc, epoch)
                    writer.add_scalar('data/test_Mean IoU', test_mean_iou, epoch)

                    # group visual
                    # writer.add_scalars('data/dice_PZ', {'train_Dice_PZ': epoch_dice1 / epoch_iter, 'test_Dice_PZ': test_dice1,}, epoch)
                    # writer.add_scalars('data/dice_CG', {'train_Dice_CG': epoch_dice2 / epoch_iter, 'test_Dice_CG': test_dice2,}, epoch)

            writer.close()
        #print("totaltime:",times)

        if self.train_opts.is_test == True:
            self.test(self.model, test_loader, model_file_best, epoch=300, visual_att=False, visual_edge=False,
                      is_attsave=False, is_save=True)

        model_file_path = os.path.join(os.getcwd(), model_file)
        model_file_best_path = os.path.join(os.getcwd(), model_file_best)
        config_path = os.path.join(os.getcwd(), self.json_filename)
        shutil.move(model_file_path, save_path)
        shutil.move(model_file_best_path, save_path)
        shutil.move(config_path, save_path)

    def test(self, model, test_loader, model_file, epoch, visual_att, visual_edge, is_attsave, is_save):
        model.load_state_dict(torch.load(model_file))
        model.eval()

        image_dir = join(self.ds_path, 'test', 'image')
        image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir)])
        lab_dir = join(self.ds_path, 'test', 'label')
        lab_filenames = sorted([join(lab_dir, x) for x in listdir(lab_dir)])
        re_image_dir = join(self.ds_path, 'test', 'reimage')
        re_image_filenames = sorted([join(re_image_dir, x) for x in listdir(re_image_dir)])

        total_dice1, total_dice2 = 0, 0
        total_mean_acc = 0
        total_mean_iou = 0

        dice1, dice2 = 0, 0
        tt_mean_acc, tt_mean_iou = 0, 0
        ASD, mSDE, HDE = [0, 0], [0, 0], [0, 0]

        per_dice1, per_dice2 = [], []
        per_mean_acc, per_mean_iou = [], []
        per_ASD1, per_ASD2, per_HDE1, per_HDE2, per_mSDE1, per_mSDE2 = [], [], [], [], [], []
        per_ppv1, per_ppv2, per_sen1, per_sen2 = [], [], [], []

        data = load_csv(self.csv_name)
        # print('test image is saving')
        for iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
            start_time = time.time()
            images, labels = images.cuda(), labels.cuda()
            prediction = model(images)

            '''
            # If it's a 5D array then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = images.size()
            if len(bs) > 4:
                inputs = images.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])
            '''
            name = os.path.basename(image_filenames[iter - 1])

            n_class = prediction.size(1)
            Overall_Acc, mean_acc, mean_iou, dice_score, ppv, sen = segmentation_stats(prediction, labels)
            total_dice1 += dice_score[1]
            total_dice2 += dice_score[2]
            total_mean_iou += mean_acc
            total_mean_acc += mean_iou

            '''
            if visual_att==True and iter == 2:
                layer_name = 'attentionblock1'
                saveSingleHeatmap(model, layer_name, images, epoch=epoch, iter=iter, filename = self.model_type)

            if visual_edge == True and iter ==2:
                logits_edge = F.softmax(prediction, dim=1)
                pred_seg_edge = logits_edge[0].data.max(0)[1].cpu().numpy()
                wdis_map = tensor_weight_distance_metric_norm(pred_seg_edge, labels[0][0].cpu().numpy(),self.model_opts.input_nc)
                wmap_visual = np.zeros_like(wdis_map[:, :, 0])
                for d in range(48):
                    weight_map = wdis_map[:, :, d]
                    weight_map = weight_map ** 2
                    wmap_visual += weight_map
                wmap_visual = np.sqrt(wmap_visual)
                min = np.amin(wmap_visual)
                max = np.amax(wmap_visual)
                wmap_visual = (wmap_visual - min) / (max - min + 0.000001)
                heatmap1 = cv2.applyColorMap((wmap_visual * 255).astype(np.uint8), cv2.COLORMAP_JET)
                save_path = 'visual/edge/{}'.format(self.model_type)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, 'epoch_{}.jpg'.format(epoch)), heatmap1)
            '''

            if is_save == True:
                # ori data
                re_ori = np.load(re_image_filenames[iter - 1])
                re_ori_shape = re_ori.shape

                gt = sitk.ReadImage(lab_filenames[iter - 1])
                gt_array = sitk.GetArrayFromImage(gt)

                #gt_array[np.where(gt_array == 2)] = 1

                # gt_array[np.where(gt_array == 3)] = 0
                # gt_array[np.where(gt_array == 4)] = 0

                direction = gt.GetDirection()
                #print(direction[0],direction[4],direction[8])
                if direction[0] < 0:
                    gt_array = gt_array[:, :, ::-1]
                if direction[4] < 0:
                    gt_array = gt_array[:, ::-1, :]
                if direction[8] < 0:
                    gt_array = gt_array[::-1, :, :]

                gt_array = np.transpose(gt_array, (1, 2, 0))
                # gt_array = fillHole_3D(gt_array)

                # prediction
                logits = F.softmax(prediction, dim=1)

                pred_seg = logits[0].data.max(0)[1]  # 128,128,48

                # crop
                trans = ts.Compose([  # ts.ToTensor(),
                    ts.AddChannel(axis=0),
                    ts.SpecialCrop(size=(self.val_scale_size[0], self.val_scale_size[1], re_ori_shape[2]), crop_type=0),
                    # ts.TypeCast(['float', 'long'])
                ])

                # re_size
                xup = int(data[iter - 1][0])
                yup = int(data[iter - 1][1])
                zup = int(data[iter - 1][2])
                if re_ori_shape[2] < self.val_scale_size[2]:
                    crop_pred_seg = trans(pred_seg)
                    crop_pred_seg = torch.squeeze(crop_pred_seg).cpu().numpy()
                    pad_pred_seg = np.pad(crop_pred_seg, ((xup, (re_ori_shape[0] - xup - self.val_scale_size[0])),
                                                          (yup, (re_ori_shape[1] - yup - self.val_scale_size[1])),
                                                          (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                else:
                    pad_pred_seg = np.pad(pred_seg.cpu().numpy(), (
                    (xup, (re_ori_shape[0] - xup - self.val_scale_size[0])),
                    (yup, (re_ori_shape[1] - yup - self.val_scale_size[1])),
                    (zup, (re_ori_shape[2] - zup - self.val_scale_size[2]))), 'constant',
                                          constant_values=((0, 0), (0, 0), (0, 0)))

                # un-resample

                new_spacing = np.array(list(gt.GetSpacing()))

                spacing = [0.625, 0.625, 1.5]

                predic = resample(pad_pred_seg, spacing, new_spacing, is_seg=True, order=1, cval=-1,
                                  dtype_data=np.uint8)

                end_time = time.time()
                print(iter, 'time:', end_time - start_time)

                dice, ppv, sen = dice_score_def(gt_array, predic, n_class)
                dice1 += dice[1]
                dice2 += dice[2]
                per_dice1.append(dice[1])
                per_dice2.append(dice[2])
                per_ppv1.append(ppv[1])
                per_ppv2.append(ppv[2])
                per_sen1.append(sen[1])
                per_sen2.append(sen[2])

                _, macc, _, miou = segmentation_scores(gt_array, predic, n_class)
                tt_mean_acc += macc
                tt_mean_iou += miou
                per_mean_acc.append(macc)
                per_mean_iou.append(miou)

                for i in range(n_class - 1):
                    dis = distance_metric(gt_array, predic, dx=new_spacing[0], k=i + 1)

                    mSDE[i] += dis[0]
                    HDE[i] += dis[1]
                    ASD[i] += dis[2]

                    if i == 0:
                        per_mSDE1.append(dis[0])
                        per_HDE1.append(dis[1])
                        per_ASD1.append(dis[2])
                    else:
                        per_mSDE2.append(dis[0])
                        per_HDE2.append(dis[1])
                        per_ASD2.append(dis[2])

                img_save = np.transpose(predic, (2, 0, 1))
                if direction[0] < 0:
                    img_save = img_save[:, :, ::-1]
                if direction[4] < 0:
                    img_save = img_save[:, ::-1, :]
                if direction[8] < 0:
                    img_save = img_save[::-1, :, :]


                img_save = sitk.GetImageFromArray(img_save)
                img_save.SetOrigin(gt.GetOrigin())
                img_save.SetSpacing(gt.GetSpacing())
                img_save.SetDirection(gt.GetDirection())
                img_save = sitk.Cast(img_save, sitk.sitkUInt8)
                name = os.path.basename(image_filenames[iter - 1])

                save_path = './prediction_boundary/{}/'.format(self.model_type)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                if name.split('.')[-1] == 'nrrd':
                    sitk.WriteImage(img_save, os.path.join(save_path, '{}.nrrd'.format(name[:-5])))
                elif name.split('.')[-1] == 'nii':
                    sitk.WriteImage(img_save, os.path.join(save_path, '{}.nii'.format(name[:-4])))
                else:
                    sitk.WriteImage(img_save, os.path.join(save_path, '{}.nii.gz'.format(name[:-7])))


        if is_save == True:
            print("dice_score1", dice1 / iter, "dice_score2", dice2 / iter)
            print("per_dice1:", per_dice1, "per_dice2:", per_dice2)
            print("mSDE1:", mSDE[0] / iter, "HDE1:", HDE[0] / iter,"ASD1:", ASD[0] / iter)
            print("mSDE2:", mSDE[1] / iter, "HDE2:", HDE[1] / iter,"ASD2:", ASD[1] / iter)
            print("per_mSDE1", per_mSDE1, "per_mSDE2", per_mSDE2)
            print("per_HDE1", per_HDE1, "per_HDE2", per_HDE2)
            print("mean_acc:", tt_mean_acc / iter, "mean_iou:", tt_mean_iou / iter)
            print("per_mean_acc:", per_mean_acc, "per_mean_iou:", per_mean_iou)

            import_data = {"dice1": per_dice1, "dice2": per_dice2,"ASD1": per_ASD1, "ASD2":per_ASD2,
                           'HDE1': per_HDE1, 'HDE2': per_HDE2, 'MDE1': per_mSDE1, 'MDE2': per_mSDE2,
                           'mean_acc': per_mean_acc, 'mean_iou': per_mean_iou, "ppv1": per_ppv1, "ppv2": per_ppv2,
                           "sen1": per_sen1, "sen2": per_sen2}
            save_path = os.path.join(os.getcwd(), 'save/seg/boundary', self.model_type, '{}.csv'.format(self.model_type))
            save_results(save_path=save_path, import_data=import_data)

        return total_dice1 / iter, total_dice2 / iter, total_mean_acc / iter, total_mean_iou / iter

def save_results(save_path, import_data):
    save_file_path = save_path

    col_name = list(import_data.keys())
    col_len = len(import_data[col_name[0]])

    with open(save_file_path, "w", encoding="utf-8") as fw:
        first_line = ",".join(col_name)
        fw.write(first_line + "\n")
        for i in range(col_len):
            row = []
            for col in col_name:
                row.append(str(import_data[col][i]))
            rows = ",".join(row)
            fw.write(rows + '\n')
        fw.closed


if __name__ == '__main__':

    # from main import train
    configs = ['config.json']
    for i in range(len(configs)):
        m = main_seg(configs[i])
        m.train()