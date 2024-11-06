import cv2
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.utils import read
import tifffile as tf
import glob

cv2.setNumThreads(0)
   
class ParticleDataset(Dataset):
    def __init__(self, mode, data_root, dataset_name, snr, density, npoints):
        self.npoints = npoints
        self.dataset_name = dataset_name
        self.mode = mode
        self.data_root = data_root
        self.pc_root = os.path.join(self.data_root, 'particle-flow', dataset_name)
        self.dataset_class = self.dataset_name+' snr '+str(snr)+' density '+density
        

    def getimg(self, index):
        img_list = []
        flow_list = []
        data_inx = index // 1998 + 1
        if self.mode == "train":
            dataset_class = self.dataset_class + ' ' + str(data_inx)
            img_path = os.path.join(self.data_root, self.dataset_name, dataset_class)
            flow_path = os.path.join(self.data_root, 'flow', self.dataset_name, dataset_class)
            imgs = sorted(os.listdir(img_path))
            flows = sorted(os.listdir(flow_path))
            imgs = [im for im in imgs if im.endswith('tif')]
            flows = [fl for fl in flows if fl.endswith('flo')]
            img_list.append(os.path.join(img_path, imgs[index%1998])) # I0
            img_list.append(os.path.join(img_path, imgs[index%1998+1])) # I1
            img_list.append(os.path.join(img_path, imgs[index%1998+2])) # I2
            flow_list.append(os.path.join(flow_path, flows[(index%1998)*2])) # flow0
            flow_list.append(os.path.join(flow_path, flows[(index%1998)*2+1])) # flow1

            # Load images
            img0 = tf.imread(img_list[0])
            gt = tf.imread(img_list[1])
            img1 = tf.imread(img_list[2])
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            flow0 = read(flow_list[0])
            flow1 = read(flow_list[1])
            flow = np.concatenate((flow0, flow1), 2).astype(np.float64)
            return img0, gt, img1, flow
        else:
            dataset_class = self.dataset_class
            img_path = os.path.join(self.data_root, self.dataset_name, dataset_class)
            imgs = sorted(os.listdir(img_path))
            imgs = [im for im in imgs if im.endswith('tif')]
            img_list.append(os.path.join(img_path, imgs[index%1998])) # I0
            img_list.append(os.path.join(img_path, imgs[index%1998+1])) # I1
            img_list.append(os.path.join(img_path, imgs[index%1998+2])) # I2
            # Load images
            img0 = tf.imread(img_list[0])
            gt = tf.imread(img_list[1])
            img1 = tf.imread(img_list[2])
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            return img0, gt, img1 


    def __getitem__(self, index):
        data_inx = index // 1998 + 1
        if self.mode == "train":
            dataset_class = self.dataset_class + ' ' + str(data_inx)
        else:
            dataset_class = self.dataset_class
        pcpath = glob.glob(os.path.join(self.pc_root, dataset_class, '*.npz'))
        pcpath = [d for d in pcpath]
        pcpath.sort()
        
        fn = pcpath[index%1998]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos0 = data['pos1'].astype('float32')
            pos1 = data['pos2'].astype('float32')

        n0 = pos0.shape[0]
        n1 = pos1.shape[0]
        pad_len0 = self.npoints - n0
        pad_len1 = self.npoints - n1

        while pad_len0>n0:
            pl0 = min(pad_len0, n0)
            pad_len0 = pad_len0-n0
            pos0 = np.concatenate((pos0, pos0[:pl0])) # N*2
        while pad_len1>n1:
            pl1 = min(pad_len1, n1)
            pad_len1 = pad_len1-n1
            pos1 = np.concatenate((pos1, pos1[:pl1]))

        pos0 = torch.from_numpy(np.concatenate((pos0, pos0[:pad_len0]))) # N*2
        pos1 = torch.from_numpy(np.concatenate((pos1, pos1[:pad_len1])))

        if self.mode == 'train':
            img0, imgt, img1, flow = self.getimg(index)   # 
            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
            embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
            return {'pos0': pos0, 'pos1': pos1, 'n0': n0, 'n1': n1, 'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'flow': flow.float(), 'embt': embt} # 

        else:
            img0, imgt, img1 = self.getimg(index)
            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

            return {'pos0': pos0, 'pos1': pos1, 'n0': n0, 'n1': n1, 'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'embt': embt}

    def __len__(self):
        if self.mode == "train":
            return 5994
        else:
            return 98


class ParticleDatasetReal(Dataset):
    def __init__(self, mode, data_root, dataset_name, npoints):
        self.npoints = npoints
        self.dataset_name = dataset_name
        self.mode = mode
        self.data_root = data_root
        self.pcpath = glob.glob(os.path.join(data_root, 'particle-flow', dataset_name, mode, '*.npz'))
        self.pcpath = [d for d in self.pcpath]
        self.pcpath.sort()

    def getimg(self, index):
        img_list = []
        flow_list = []
        if self.mode == "train":
            img_path = os.path.join(self.data_root, self.dataset_name, self.mode)
            flow_path = os.path.join(self.data_root, 'flow', self.dataset_name)
            imgs = sorted(os.listdir(img_path))
            flows = sorted(os.listdir(flow_path))
            imgs = [im for im in imgs if im.endswith('tif')]
            flows = [fl for fl in flows if fl.endswith('flo')]
            img_list.append(os.path.join(img_path, imgs[index])) # I0
            img_list.append(os.path.join(img_path, imgs[index+1])) # I1
            img_list.append(os.path.join(img_path, imgs[index+2])) # I2
            flow_list.append(os.path.join(flow_path, flows[(index)*2])) # flow0
            flow_list.append(os.path.join(flow_path, flows[(index)*2+1])) # flow1

            # Load images
            img0 = tf.imread(img_list[0])
            gt = tf.imread(img_list[1])
            img1 = tf.imread(img_list[2])
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            flow0 = read(flow_list[0])
            flow1 = read(flow_list[1])
            flow = np.concatenate((flow0, flow1), 2).astype(np.float64)
            return img0, gt, img1, flow    
        else:
            img_path = os.path.join(self.data_root, self.dataset_name, self.mode)
            imgs = sorted(os.listdir(img_path))
            imgs = [im for im in imgs if im.endswith('tif')]
            img_list.append(os.path.join(img_path, imgs[index])) # I0
            img_list.append(os.path.join(img_path, imgs[index+1])) # I1
            img_list.append(os.path.join(img_path, imgs[index+2])) # I2

            # Load images
            img0 = tf.imread(img_list[0])
            gt = tf.imread(img_list[1])
            img1 = tf.imread(img_list[2])
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            return img0, gt, img1


    def __getitem__(self, index):
        fn = self.pcpath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos0 = data['pos1'].astype('float32')
            pos1 = data['pos2'].astype('float32')

        n0 = pos0.shape[0]
        n1 = pos1.shape[0]
        pad_len0 = self.npoints - n0
        pad_len1 = self.npoints - n1

        while pad_len0>n0:
            pl0 = min(pad_len0, n0)
            pad_len0 = pad_len0-n0
            pos0 = np.concatenate((pos0, pos0[:pl0])) # N*2
        while pad_len1>n1:
            pl1 = min(pad_len1, n1)
            pad_len1 = pad_len1-n1
            pos1 = np.concatenate((pos1, pos1[:pl1]))

        pos0 = torch.from_numpy(np.concatenate((pos0, pos0[:pad_len0]))) # N*2
        pos1 = torch.from_numpy(np.concatenate((pos1, pos1[:pad_len1])))

        if self.mode == 'train':
            img0, imgt, img1, flow = self.getimg(index)  
            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
            embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
            return {'pos0': pos0, 'pos1': pos1, 'n0': n0, 'n1': n1, 'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'flow': flow.float(), 'embt': embt}

        else:
            img0, imgt, img1 = self.getimg(index)
            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
            embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

            return {'pos0': pos0, 'pos1': pos1, 'n0': n0, 'n1': n1, 'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'embt': embt}

    def __len__(self):
        imgs = sorted(os.listdir(os.path.join(self.data_root, self.dataset_name, self.mode)))
        imgs = [im for im in imgs if im.endswith('tif')]
        return len(imgs)-2
