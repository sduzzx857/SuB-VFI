import os
import sys
import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn.functional as F
import cv2
import tqdm
from omegaconf import OmegaConf
sys.path.append('.')
from utils.utils import read, write
import tifffile as tf
from flow_generation.liteflownet.run import estimate


def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('--density', type=str, default='')
args = parser.parse_args()
if args.dataset in ['MICROTUBULE', 'RECEPTOR', 'VESICLE']:
    args.config = 'cfgs/' + args.dataset + '-' + args.density + '.yaml'
    cfg = OmegaConf.load(args.config)

    particle_dir = cfg.data.train.params.data_root
    dataset_name = cfg.data.train.params.dataset_name
    snr = cfg.data.train.params.snr
    density = cfg.data.train.params.density

    for i in range(1,4):
        dataset_class = dataset_name+' snr '+str(snr)+' density '+density + ' ' + str(i)
        particle_sequences_dir = osp.join(particle_dir, dataset_name, dataset_class)
        particle_flow_dir = osp.join(particle_dir, 'flow', dataset_name, dataset_class)

if args.dataset in ['EB1', 'LYSOSOME', 'CCR5']:
    args.config = 'cfgs/' + args.dataset + '.yaml'
    cfg = OmegaConf.load(args.config)

    particle_dir = cfg.data.train.params.data_root
    dataset_name = cfg.data.train.params.dataset_name

    particle_sequences_dir = osp.join(particle_dir, dataset_name, 'train')
    particle_flow_dir = osp.join(particle_dir, 'flow', dataset_name)   

print('Built Flow Path')
if not osp.exists(particle_flow_dir):
    os.makedirs(particle_flow_dir)


imgs = sorted(os.listdir(particle_sequences_dir))
imgs = [f for f in imgs if f.endswith("tif")] 
for i in tqdm.tqdm(range(len(imgs)-2)):

    img0_path = os.path.join(particle_sequences_dir, imgs[i])
    imgt_path = os.path.join(particle_sequences_dir, imgs[i+1])
    img1_path = os.path.join(particle_sequences_dir, imgs[i+2])
    flow_t0_name = imgs[i+1][:-4] + '_t0.flo'
    flow_t1_name = imgs[i+1][:-4] + '_t1.flo'
    flow_t0_path = osp.join(particle_flow_dir, flow_t0_name)
    flow_t1_path = osp.join(particle_flow_dir, flow_t1_name)

    img0 = tf.imread(img0_path)
    imgt = tf.imread(imgt_path)
    img1 = tf.imread(img1_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    imgt = cv2.cvtColor(imgt, cv2.COLOR_GRAY2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    
    flow_t0 = pred_flow(imgt, img0)
    flow_t1 = pred_flow(imgt, img1)
    
    write(flow_t0_path, flow_t0)
    write(flow_t1_path, flow_t1)