import sys
import tqdm
import torch
import numpy as np
import os.path as osp
import os
import cv2
from omegaconf import OmegaConf
import tifffile as tf
import shutil
import argparse
import xml.etree.ElementTree as ET
sys.path.append('.')
from utils.build_utils import build_from_cfg
from utils.utils import read, img2tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_particle_location(args):
    save = args.loc_path
    xml_path = os.path.join(args.img_path, '_Tracks.xml')
    if not os.path.exists(save):
        os.makedirs(save)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pos = []

    for particle in root.findall('.//particle'):
        for i, detection1 in enumerate(particle.findall('detection')):
            if i < len(particle.findall('detection')): 
                t1 = int(detection1.get('t'))
                x1 = float(detection1.get('x'))
                y1 = float(detection1.get('y'))       
                if t1 >= len(pos):
                    for j in range(t1 - len(pos) + 1):
                        pos.append([])
                pos[t1].append([x1, y1])

    print("frame num:", len(pos))
    max_num = 0
    for i in range(len(pos)):
        p = np.array(pos[i])

        # Sort vertices by y then x
        sort_inds = np.lexsort(p.T)
        p = p[sort_inds]

        file = 't%04d.npz'%(i)
        np.savez(os.path.join(save, file), pos=p)
        max_num = max(p.shape[0], max_num)
    
    print("The peak number of particles: ", max_num)

def output(model, args):
    strPath = args.img_path
    savePath = args.save_path
    pcPath = args.loc_path

    if not osp.exists(savePath):
        os.makedirs(savePath)   
    
    imglist = os.listdir(strPath)
    imglist = [f for f in imglist if f.endswith("tif")]    
    imglist = sorted(imglist)
    pclist = os.listdir(pcPath)
    pclist = [f for f in pclist if f.endswith("npz")] 
    pclist = sorted(pclist)
    pbar = tqdm.tqdm(range(0, len(imglist)-1))
    for i in pbar:
        pc1 = os.path.join(pcPath, pclist[i])
        pc2 = os.path.join(pcPath, pclist[i+1])
        with open(pc1, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos'].astype('float32')
        with open(pc2, 'rb') as fp:
            data = np.load(fp)
            pos2 = data['pos'].astype('float32')

        m1 = pos1.shape[0]
        m2 = pos2.shape[0]
        pos1 = torch.tensor(pos1).unsqueeze(0).float().to('cuda')
        pos2 = torch.tensor(pos2).unsqueeze(0).float().to('cuda')
        m1 = torch.tensor(m1).unsqueeze(0).float().to('cuda')
        m2 = torch.tensor(m2).unsqueeze(0).float().to('cuda')
        if i == 0:
            shutil.copy(osp.join(strPath, imglist[i]), os.path.join(savePath, 'image%04d.tif' % (i)))
        shutil.copy(osp.join(strPath, imglist[i+1]), os.path.join(savePath, 'image%04d.tif' % ((i+1)*2)))        
        I0 = tf.imread(osp.join(strPath, imglist[i]))
        I2 = tf.imread(osp.join(strPath, imglist[i+1]))
        I0 = cv2.cvtColor(I0, cv2.COLOR_GRAY2RGB)
        I2 = cv2.cvtColor(I2, cv2.COLOR_GRAY2RGB)
        I0 = img2tensor(I0).to(device)
        I2 = img2tensor(I2).to(device)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
        with torch.no_grad():
            pred = model(pos1, pos2, m1, m2, I0, I2, embt, scale_factor=1.0, eval=True)['imgt_pred']
        I1 = (pred[0].cpu().numpy().transpose(1,2,0)*255).round().astype('uint8')
        I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
        tf.imwrite(osp.join(savePath, 'image%04d.tif' % (2*i+1)), I1)      


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='EB1.yaml') 
parser.add_argument('--ckpt', default='EB1.pth') 
parser.add_argument('--img_path', default='demos/images') 
parser.add_argument('--loc_path', default='demos/particle-loc') 
parser.add_argument('--save_path', default='demos/results') 
args = parser.parse_args()

extract_particle_location(args)

cfg = args.cfg
cfg_path = os.path.join('cfgs',cfg)
ckpt = args.ckpt
ckpt_path = os.path.join('pretrained/', ckpt)
network_cfg = OmegaConf.load(cfg_path).network
network_cfg.params.gmsf_pretrained=''
network_cfg.params.syn_pretrained=''
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)

if os.path.exists(ckpt_path):
    print("load from: ", ckpt_path)
else:
    print("Wrong pretrained path!!!")

model.load_state_dict(ckpt['state_dict'])
model = model.to(device)
model.eval()

output(model, args)

