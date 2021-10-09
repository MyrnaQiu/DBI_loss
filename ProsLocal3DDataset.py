#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:18:33 2019

@author: myrna
"""

import torch.utils.data as data
from os.path import join
from os import listdir
import numpy as np
import scipy

class ProsLocal3DDataset(data.Dataset):
    def __init__(self, root_dir, split, zoom_shape, transform=None):
        super(ProsLocal3DDataset, self).__init__()
        
        self.split = split
        self.zoom_shape = zoom_shape
        

        image_dir = join(root_dir, split, 'reimage')
        target_dir = join(root_dir, split, 'relabel')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) ])   #01 02
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) ])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        img_arr = np.load(self.image_filenames[index]).astype(np.float32)
        target = np.load(self.target_filenames[index]).astype(np.uint8)
        
        img_shape = img_arr.shape
        target_shape = target.shape

        img_arr = scipy.ndimage.zoom(img_arr, np.array(self.zoom_shape)/np.array([img_shape[0],img_shape[1],img_shape[2]]), order=0).astype(np.float32)
        target = scipy.ndimage.zoom(target, np.array(self.zoom_shape)/np.array([target_shape[0],target_shape[1],target_shape[2]]), order=0).astype(np.uint8)
        target[np.where(target == 2)]=1
       
        if self.transform:
            img_arr, target = self.transform(img_arr, target)
            
        return img_arr, target