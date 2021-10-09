#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:40:16 2019

@author: myrna
"""

import torch.utils.data as data
from os.path import join
from os import listdir
from util import *
import random

class Pros3DDataset(data.Dataset):
    def __init__(self, root_dir, split, crop_shape, csv_name, transform=None):
        super(Pros3DDataset, self).__init__()
        self.setup_seed(1234)
        self.split = split
        self.crop_shape = crop_shape
        self.data = load_csv(csv_name)
        

        image_dir = join(root_dir, split, 'reimage')
        target_dir = join(root_dir, split, 'relabel')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) ])   #01 02
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) ])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_filenames)

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def __getitem__(self, index):
        if self.split =='test':
            img_arr = np.load(self.image_filenames[index]).astype(np.float32)
            target = np.load(self.target_filenames[index]).astype(np.uint8)
            
            xup = int(self.data[index][0])
            yup = int(self.data[index][1])
            zup = int(self.data[index][2])
            roi_img = img_arr[xup:xup+self.crop_shape[0],yup:yup+self.crop_shape[1],zup:zup+self.crop_shape[2]].astype(np.float32)
            roi_target = target[xup:xup+self.crop_shape[0],yup:yup+self.crop_shape[1],zup:zup+self.crop_shape[2]].astype(np.uint8)
            
            if self.transform:
                roi_img, roi_target = self.transform(roi_img, roi_target)
                
            return roi_img, roi_target
            
        else:
            img_arr = np.load(self.image_filenames[index]).astype(np.float32)
            target = np.load(self.target_filenames[index]).astype(np.uint8)
            
            xmin = np.min(np.where(target!=0)[0]) 
            xmax = np.max(np.where(target!=0)[0]) 
            ymin = np.min(np.where(target!=0)[1]) 
            ymax = np.max(np.where(target!=0)[1]) 
            zmin = np.min(np.where(target!=0)[2]) 
            zmax = np.max(np.where(target!=0)[2])
                
            xup=xmin-(self.crop_shape[0]-(xmax-xmin))//2 #down
            yup=ymin-(self.crop_shape[1]-(ymax-ymin))//2
            zup=zmin-(self.crop_shape[2]-(zmax-zmin))//2
            
            #if xup<0:
                #xup=0
            #if yup<0:
                #yup=0
            if zup<0:
                zup=0
                
            roi_img = img_arr[xup:xup+self.crop_shape[0],yup:yup+self.crop_shape[1],zup:zup+self.crop_shape[2]].astype(np.float32)
            roi_target = target[xup:xup+self.crop_shape[0],yup:yup+self.crop_shape[1],zup:zup+self.crop_shape[2]].astype(np.uint8)

            #print(self.image_filenames[index],self.target_filenames[index])
            if self.transform:
                roi_img, roi_target = self.transform(roi_img, roi_target)
                
            return roi_img, roi_target