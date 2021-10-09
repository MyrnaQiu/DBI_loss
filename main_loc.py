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
import torch.nn as nn
import os
from os.path import join
from os import listdir
import numpy as np
import torchsample.transforms as ts
from ProsLocal3DDataset import ProsLocal3DDataset
from tqdm import tqdm

from model.unet3d_dsv_ag_att import unet3d_dsv_ag_att
from model.unet3d_dsv_noagatt import unet3d_dsv_noagatt


from model.networks_other import print_model

from loss import *

from util import *
from post_process_crf import apply_crf
from tensorboardX import SummaryWriter
from skimage import morphology,measure
import shutil
import scipy
import SimpleITK as sitk

class main_loc():
    def __init__(self,json_filename):
        super(main_loc, self).__init__()
        self.json_filename = json_filename
        # Load options
        self.json_opts = json_file_to_pyobj(self.json_filename)
        self.train_opts = self.json_opts.training
        self.model_opts = self.json_opts.model
        self.model_type = self.json_opts.model.model_type
        self.model_name = self.json_opts.model.model_name
        self.model_pretype = self.json_opts.model.model_pretype
        self.zoom_shape = self.json_opts.augmentation.pros.scale_size
        self.ds_path = self.json_opts.path.data_path
        self.crop_shape = self.json_opts.augmentation.pros.crop_size
        self.patience = self.json_opts.model.patience
        self.sum_type = self.json_opts.model.sum_type
        self.e_weight = self.json_opts.model.e_weight

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
                self.criterion = DiceFocalLoss(n_classes=self.model_opts.output_nc, weight=0.5, gamma=4,
                                               alpha=[0.1, 0.6, 0.3], size_average=True)
            elif self.json_opts.model.criterion == 'dice':
                self.criterion = SoftDiceLoss(self.model_opts.output_nc)
            elif self.json_opts.model.criterion == 'focal':
                self.criterion = FocalLoss(gamma=2, alpha=[1, 6, 3], size_average=True)
            elif self.json_opts.model.criterion == 'ce':
                self.criterion = cross_entropy_3D(alpha=None, size_average=True)
            elif self.json_opts.model.criterion == 'dice_ce':
                self.criterion = Dice_cross_entropy_3D(self.model_opts.output_nc, alpha=None, size_average=True)
            elif self.json_opts.model.criterion == 'dbi_ce':
                self.criterion = dbi_cross_entropy_3D(self.model_opts.output_nc,size_average=True)
            elif self.json_opts.model.criterion == 'dbi_focal':
                self.criterion = dbi_focal_3D(self.model_opts.output_nc,gamma=2, size_average=True)
            elif self.json_opts.model.criterion == 'dice_dbi_ce':
                self.criterion = Softdice_dbi_cross_entropy_3D(self.model_opts.output_nc, self.e_weight,
                                                               size_average=True)
            elif self.json_opts.model.criterion == 'dice_dbi_focal':
                self.criterion = Softdice_dbiFocal_loss(self.model_opts.output_nc, self.e_weight, gamma=2,
                                                        size_average=True)

            self.optimizer = self.get_optimizer(self.model_opts.optimizer_opt, filter(lambda p: p.requires_grad, self.model.parameters()))


    def get_optimizer(self,option, params):
    #opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
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
        arch_type = self.train_opts.arch_type  #"arch_type": "pros":

        model_file = '{}.pkl'.format(self.model_type)
        model_file_best = '{}_best.pkl'.format(self.model_type)

        save_path = os.path.join(os.getcwd(),'save/loc',self.model_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Setup Dataset and Augmentation
        ds_transform = get_dataset_transformation(arch_type, opts=self.json_opts.augmentation)

         # Setup Data Loader
        train_dataset = ProsLocal3DDataset(self.ds_path, split='train', zoom_shape=self.zoom_shape,
                                           transform=ds_transform['train'])
        test_dataset = ProsLocal3DDataset(self.ds_path, split='test', zoom_shape=self.zoom_shape,
                                          transform=ds_transform['valid'])

        train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=self.train_opts.batchSize, shuffle=True)
        test_loader  = DataLoader(dataset=test_dataset,  num_workers=0, batch_size= 1, shuffle=False)

        num_params = print_model(self.model)

        if self.train_opts.is_train:

            bestmodel = 10
            best_epoch = 0
            total_epoch = self.train_opts.n_epochs
            writer = SummaryWriter('runs/loc/{}'.format(self.model_type))
            stop_num = 0
            #self.set_scheduler(self.train_opts)

            early_stop  = False

            if self.train_opts.is_pretrain == True:
                self.model.load_state_dict((torch.load(model_file_best)))

            for epoch in range(total_epoch):
                if not early_stop:

                    epoch_loss = 0
                    epoch_dice1 = 0

                    # Training Iterations
                    for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                        images , labels = images.cuda(), labels.cuda()
                        #print(labels.shape)#[batchsize, 1, 160, 160, 48]
                        # Forward + Backward + Optimize

                        self.optimizer.zero_grad()
                        outputs = self.model(images)

                        loss = self.criterion(outputs, labels)

                        epoch_loss += loss.item()
                        loss.backward()
                        self.optimizer.step()

                        _, _, _, dice_score,_,_ = segmentation_stats(outputs, labels)
                        epoch_dice1 += dice_score[1]

                    # epoch_end = time.perf_counter()
                    print("Epoch [%d/%d], train_Loss: %.4f" % (epoch + 1, total_epoch, epoch_loss / epoch_iter))
                    print("dice_score1:%.4f " % (epoch_dice1 / epoch_iter))

                    ##save parameters
                    torch.save(self.model.state_dict(), model_file)

                    #print('the epoch takes time:',epoch_end-epoch_start)

                    if epoch_loss/epoch_iter <=bestmodel:
                        torch.save(self.model.state_dict(), model_file_best)
                        bestmodel = epoch_loss/epoch_iter
                        best_epoch = epoch+1
                        stop_num = 0
                    else:
                        stop_num += 1

                    print('best epoch:',best_epoch)
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
                    test_dice1 = self.test(self.model, test_loader, model_file, is_csv=False)
                    print("test_dice_score1:%.4f " % (test_dice1))

                    ##visual
                    writer.add_scalar('data/train_loss', epoch_loss / epoch_iter, epoch)
                    writer.add_scalar('data/train_Dice Coefficient1', epoch_dice1 / epoch_iter, epoch)

            writer.close()


        if self.train_opts.is_test == True:
            self.test(self.model, test_loader, model_file_best, is_csv= True)

        model_file_path = os.path.join(os.getcwd(), model_file)
        model_file_best_path = os.path.join(os.getcwd(), model_file_best)
        config_path = os.path.join(os.getcwd(), self.json_filename)
        shutil.copy(model_file_path, save_path)
        shutil.copy(model_file_best_path, save_path)
        shutil.copy(config_path, save_path)

    def test(self, model, test_loader, model_file, is_csv= False):
         model.load_state_dict(torch.load(model_file))
         model.eval()

         image_dir = join(self.ds_path, 'test', 'image')
         image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) ])
         re_image_dir = join(self.ds_path, 'test', 'reimage')
         re_image_filenames  = sorted([join(re_image_dir, x) for x in listdir(re_image_dir) ])
         re_label_dir = join(self.ds_path, 'test', 'relabel')
         re_label_filenames = sorted([join(re_label_dir, x) for x in listdir(re_label_dir)])

         total_dice1 = 0

         for iter, (images, labels) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):

            images , labels =images.cuda(), labels.cuda()
            prediction = model(images)

            Overall_Acc,mean_acc, mean_iou, dice_score,_,_ = segmentation_stats(prediction, labels)
            total_dice1 += dice_score[1]

            # save csv
            if is_csv == True:
                logits = F.softmax(prediction, dim=1)
                pred_seg = logits[0].data.max(0)[1].cpu()

                '''
                re_range = (np.array(self.zoom_shape) - np.array(self.patch_size))/2
                re_range = re_range.astype(np.int)
                re_pred_seg = np.pad(pred_seg,  ((re_range[0], re_range[0]), (re_range[1], re_range[1]), (0,0)), mode='constant')
                '''
                ## re_size
                target = np.load(re_label_filenames[iter - 1]).astype(np.uint8)
                image = np.load(re_image_filenames[iter - 1]).astype(np.float32)
                ori = sitk.ReadImage(image_filenames[iter - 1])

                target_shape = target.shape
                #print(target_shape)
                resize_pred_seg = scipy.ndimage.zoom(pred_seg, np.array(
                    [target_shape[0], target_shape[1], target_shape[2]]) / np.array(self.zoom_shape), order=0)
                resize_pred_seg = resize_pred_seg.astype(np.ubyte)

                # post operation
                pre_op = measure.label(resize_pred_seg, connectivity=3)
                props = measure.regionprops(pre_op)
                if len(props) > 1:
                    resize_pred_seg = morphology.remove_small_objects(pre_op, min_size=3000, connectivity=3,
                                                                      in_place=False)
                resize_pred_seg = resize_pred_seg.astype(np.ubyte)

                xmin = np.min(np.where(resize_pred_seg != 0)[0])
                xmax = np.max(np.where(resize_pred_seg != 0)[0])
                ymin = np.min(np.where(resize_pred_seg != 0)[1])
                ymax = np.max(np.where(resize_pred_seg != 0)[1])
                zmin = np.min(np.where(resize_pred_seg != 0)[2])
                zmax = np.max(np.where(resize_pred_seg != 0)[2])

                xdif = self.crop_shape[0]
                ydif = self.crop_shape[1]
                zdif = self.crop_shape[2]

                xup = xmin - (xdif - (xmax - xmin)) // 2  # down
                yup = ymin - (ydif - (ymax - ymin)) // 2
                zup = zmin - (zdif - (zmax - zmin)) // 2
                if xup < 0:
                    xup = 0
                if yup < 0:
                    yup = 0
                if zup < 0:
                    zup = 0
                if zup + zdif >= target_shape[2]:
                    zup = 0

                name = os.path.basename(image_filenames[iter - 1])

                reori_save = target[xup:xup + xdif, yup:yup + ydif, zup:zup + zdif]

                oriimg_save = image[xup:xup + xdif, yup:yup + ydif, zup:zup + zdif]


                direction = ori.GetDirection()
                if direction[0] < 0:
                    reori_save = reori_save[::-1, :, :]
                    oriimg_save = oriimg_save[::-1, :, :]
                if direction[4] < 0:
                    reori_save = reori_save[:, ::-1, :]
                    oriimg_save = oriimg_save[:, ::-1, :]
                if direction[8] < 0:
                    reori_save = reori_save[:, :, ::-1]
                    oriimg_save = oriimg_save[:, :, ::-1]

                reori_save = np.transpose(reori_save, (2, 0, 1))
                oriimg_save = np.transpose(oriimg_save, (2, 0, 1))


                reori_save = sitk.GetImageFromArray(reori_save)
                reori_save.SetOrigin(ori.GetOrigin())
                reoriimg_save = sitk.GetImageFromArray(oriimg_save)
                reoriimg_save.SetOrigin(ori.GetOrigin())



                reori_save.SetSpacing([0.625, 0.625, 1.5])
                reoriimg_save.SetSpacing([0.625, 0.625, 1.5])

                reori_save.SetDirection(ori.GetDirection())
                reoriimg_save.SetDirection(ori.GetDirection())

                if name.split('.')[-1] == 'nrrd':
                    sitk.WriteImage(reori_save,
                                    './label/Loc{}.nrrd'.format(name[:-5]))
                    sitk.WriteImage(reoriimg_save,
                                    './image/Loc{}.nrrd'.format(name[:-5]))
                elif name.split('.')[-1] == 'nii':
                    sitk.WriteImage(reori_save,
                                    './label/Loc{}.nii'.format(name[:-4]))
                    sitk.WriteImage(reoriimg_save,
                                    './image/Loc{}.nii'.format(name[:-4]))
                else:
                    sitk.WriteImage(reori_save,
                                    './label/Loc{}.nii.gz'.format(name[:-7]))
                    sitk.WriteImage(reoriimg_save,
                                    './image/Loc{}.nii.gz'.format(name[:-7]))

                # CSV write
                location = [xup, yup, zup]
                out = open('Location.csv', 'a', newline='')
                csv_write = csv.writer(out, dialect='excel')
                csv_write.writerow(location)
         return total_dice1 / len(test_loader)

if __name__ == '__main__':

    #from main import train
    m=main_loc('config_loc.json')
    m.train()