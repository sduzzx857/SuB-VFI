#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tifffile as tf
import torch
import os
import glob
import numpy as np


class Particle_Flow():
    def __init__(self, root, dataset_name, snr, density, npoints, train=True):
        self.npoints = npoints
        self.dataset_name = dataset_name
        self.train = train
        if train==True:
            self.data_root = os.path.join(root, 'train')
        else:
            self.data_root = os.path.join(root, 'test')
        self.dataset_class = self.dataset_name+' snr '+str(snr)+' density '+density
        self.pcpath = glob.glob(os.path.join(self.data_root, dataset_name, self.dataset_class, '*.npz'))
        # print(self.pcpath)
        self.pcpath = [d for d in self.pcpath]
        self.pcpath.sort()


    def __getitem__(self, index):
        data_dict = {'index': index}
        fn = self.pcpath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos1'].astype('float32')
            pos2 = data['pos2'].astype('float32')
            flow = data['flow'].astype('float32')

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        # 模拟场景流的数据分布
        pos1 = pos1 / 8
        pos2 = pos2 / 8
        flow = flow / 8

        # pos1 = pos1[:self.npoints]
        # pos2 = pos2[:self.npoints]
        # flow = flow[:self.npoints]


        n = pos1.shape[0]
        pad_len = self.npoints - n

        while pad_len>n:
            pl = min(pad_len, n)
            pad_len = pad_len-n
            pos1 = np.concatenate((pos1, pos1[:pl])) # N*2
            pos2 = np.concatenate((pos2, pos2[:pl]))
            flow = np.concatenate((flow, flow[:pl]))

        pos1 = np.concatenate((pos1, pos1[:pad_len])) # N*2
        pos2 = np.concatenate((pos2, pos2[:pad_len]))
        flow = np.concatenate((flow, flow[:pad_len])) 

        mask1 = np.zeros_like(pos1[:, 1])

        pcs = np.concatenate([pos1, pos2], axis=1)
        pcs = torch.from_numpy(pcs).float()
        flow_3d = torch.from_numpy(flow).permute(1, 0).float()
        occ_mask_3d = torch.from_numpy(mask1)

        data_dict['pcs'] = pcs
        data_dict['flow_3d'] = flow_3d
        data_dict['occ_mask_3d'] = occ_mask_3d

        return data_dict

    def __len__(self):
        return len(self.pcpath)
