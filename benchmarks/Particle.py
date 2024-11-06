import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
import os
from omegaconf import OmegaConf
import cv2
import tifffile as tf
import shutil
sys.path.append('.')
from utils.utils import read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulted_eval(model, dataset, snr, density, root):
    psnr_list = []
    ssim_list = []
    savePath = os.path.join("output/benchmarks", dataset+density)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    strPath = os.path.join(root, 'SIMULATED/test/', dataset, dataset+' snr '+snr+' density '+density)
    pcPath = os.path.join(root, 'SIMULATED/test/particle-flow', dataset, dataset+' snr '+snr+' density '+density)
    imglist = os.listdir(strPath)
    imglist = [f for f in imglist if f.endswith("tif")]    
    imglist = sorted(imglist)
    pclist = os.listdir(pcPath)
    pclist = [f for f in pclist if f.endswith("npz")] 
    pclist = sorted(pclist)
    pbar = tqdm.tqdm(range(0, len(imglist)-2 ))
    for i in pbar:

        pc = os.path.join(pcPath, pclist[i])
        with open(pc, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos1'].astype('float32')
            pos2 = data['pos2'].astype('float32')

        m1 = pos1.shape[0]
        m2 = pos2.shape[0]
        pos1 = torch.tensor(pos1).unsqueeze(0).float().to('cuda')
        pos2 = torch.tensor(pos2).unsqueeze(0).float().to('cuda')
        m1 = torch.tensor(m1).unsqueeze(0).float().to('cuda')
        m2 = torch.tensor(m2).unsqueeze(0).float().to('cuda')
        if i == 0:
            shutil.copy(os.path.join(strPath, imglist[i]), savePath)
        if i == len(imglist)-3:
            shutil.copy(os.path.join(strPath, imglist[i+2]), savePath)
        I0 = tf.imread(osp.join(strPath, imglist[i]))
        I1 = tf.imread(osp.join(strPath, imglist[i+1]))
        I2 = tf.imread(osp.join(strPath, imglist[i+2]))
        I0 = cv2.cvtColor(I0, cv2.COLOR_GRAY2RGB)
        I1 = cv2.cvtColor(I1, cv2.COLOR_GRAY2RGB)
        I2 = cv2.cvtColor(I2, cv2.COLOR_GRAY2RGB)
        I0 = img2tensor(I0).to(device)
        I1 = img2tensor(I1).to(device)
        I2 = img2tensor(I2).to(device)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
        with torch.no_grad():
            I1_pred = model(pos1, pos2, m1, m2, I0, I2, embt,
                            scale_factor=1.0, eval=True)['imgt_pred']
        est_img = (I1_pred.permute(0,2,3,1).detach().cpu().numpy()[0] * 255).round().astype('uint8')
        tf.imwrite(os.path.join(savePath, imglist[i+1]), est_img)

        psnr = calculate_psnr(I1_pred, I1) # .detach().cpu().numpy()
        ssim = calculate_ssim(I1_pred, I1) # .detach().cpu().numpy()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        desc_str = f'[{dataset}] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
        pbar.set_description_str(desc_str)

           
def real_eval(model, dataset, root):
        psnr_list = []
        ssim_list = []

        strPath = os.path.join(root, 'REAL/', dataset, 'test')
        savePath = os.path.join('output/benchmarks', dataset)
        pcPath = os.path.join(root, 'REAL/particle-flow', dataset, 'test')
        if not os.path.exists(savePath):
            os.makedirs(savePath)    
        imglist = os.listdir(strPath)
        imglist = [f for f in imglist if f.endswith("tif") or f.endswith("png")]    
        imglist = sorted(imglist)
        pclist = os.listdir(pcPath)
        pclist = [f for f in pclist if f.endswith("npz")] 
        pclist = sorted(pclist)
        pbar = tqdm.tqdm(range(0, len(imglist)-2, 2))#
        for i in pbar:

            pc = os.path.join(pcPath, pclist[i])
            with open(pc, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
            m1 = pos1.shape[0]
            m2 = pos2.shape[0]
            pos1 = torch.tensor(pos1).unsqueeze(0).float().to('cuda')
            pos2 = torch.tensor(pos2).unsqueeze(0).float().to('cuda')
            m1 = torch.tensor(m1).unsqueeze(0).float().to('cuda')
            m2 = torch.tensor(m2).unsqueeze(0).float().to('cuda')
            if i == 0:
                shutil.copy(os.path.join(strPath, imglist[i]), savePath)
            shutil.copy(os.path.join(strPath, imglist[i+2]), savePath)  
            I0 = cv2.imread(osp.join(strPath, imglist[i]))
            I1 = cv2.imread(osp.join(strPath, imglist[i+1]))
            I2 = cv2.imread(osp.join(strPath, imglist[i+2]))
            I0 = img2tensor(I0).to(device)
            I1 = img2tensor(I1).to(device)
            I2 = img2tensor(I2).to(device)
            embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
            with torch.no_grad():
                I1_pred = model(pos1, pos2, m1, m2, I0, I2, embt, scale_factor=1.0, eval=True)['imgt_pred']

            psnr = calculate_psnr(I1_pred, I1) # .detach().cpu().numpy()
            ssim = calculate_ssim(I1_pred, I1) # .detach().cpu().numpy()

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            desc_str = f'[{dataset}] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
            pbar.set_description_str(desc_str)

            I1_ = (I1_pred[0].cpu().numpy().transpose(1,2,0)*255).round().astype('uint8')
            I1_ = cv2.cvtColor(I1_, cv2.COLOR_BGR2GRAY)
            tf.imwrite(os.path.join(savePath, imglist[i+1]), I1_)  


def simulate(root):
    data_sets = ['MICROTUBULE','VESICLE','RECEPTOR' ]
    densities = [ 'high','low']
    snr = '7'
    for density in densities:
        for dataset in data_sets:
            cfg = dataset+'-low.yaml'
            cfg_path = os.path.join('cfgs',cfg)
            ckpt = dataset+'-'+density+'.pth'
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

            print(dataset+' '+density)
            simulted_eval(model, dataset, snr, density, root)

def real(root):
    data_sets = ['EB1' ,'CCR5' ,'LYSOSOME']
    for dataset in data_sets:
        cfg = dataset+'.yaml'
        cfg_path = os.path.join('cfgs',cfg)
        ckpt = dataset+'.pth'
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
        
        print(dataset)
        real_eval(model, dataset, root)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='data/',) 
args = parser.parse_args()

real(args.root)
simulate(args.root)