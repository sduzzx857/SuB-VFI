import xml.etree.ElementTree as ET
import numpy as np
import os
import argparse
from math import sqrt
from skimage.feature import blob_log
import tifffile as tf

def creat_dataset(data_root, dataset_name, density, snr, inx=None):
    dataset_class = dataset_name + ' snr ' + str(snr) + ' density ' + density
    if inx==None:
        save = os.path.join(data_root, 'particle-flow', dataset_name, dataset_class)
        tree = ET.parse(os.path.join(data_root, dataset_name, dataset_class, '_Tracks.xml'))
    else: 
        save = os.path.join(data_root, 'particle-flow', dataset_name, dataset_class+' '+str(inx))
        tree = ET.parse(os.path.join(data_root, dataset_name, dataset_class+' '+str(inx), dataset_class+'.xml'))

    if not os.path.exists(save):
        os.makedirs(save)   
    root = tree.getroot()

    flow_dict = []
    pos1 = []
    pos2 = []
    flow = []

    for particle in root.findall('.//particle'):
        for i, detection1 in enumerate(particle.findall('detection')):
            if i < len(particle.findall('detection'))-2:
                t1 = int(detection1.get('t'))
                x1 = float(detection1.get('x'))
                y1 = float(detection1.get('y'))
                detection2 = particle.findall('detection')[i+2]
                x2 = float(detection2.get('x'))
                y2 = float(detection2.get('y'))      
                if t1 >= len(flow_dict):
                    for j in range(t1 - len(flow_dict) + 1):
                        flow_dict.append({})
                        pos1.append([])
                        pos2.append([])
                        flow.append([])
                flow_dict[t1][(x1, y1)] = [round(x2-x1, 3), round(y2-y1, 3)]
                pos1[t1].append([x1, y1])
                pos2[t1].append([x2, y2])
                flow[t1].append([round(x2-x1, 3), round(y2-y1, 3)])

    print("frame num:", len(pos1))
    for i in range(len(pos1)):
        f = np.array(flow[i])
        p1 = np.array(pos1[i])
        p2 = np.array(pos2[i])

        # Sort vertices by y then x
        sort_inds = np.lexsort(p1.T)
        p1 = p1[sort_inds]
        f = f[sort_inds]
        sort_inds = np.lexsort(p2.T)
        p2 = p2[sort_inds]
        if inx == None:
            file = dataset_class + ' t%03d.npz'%(i)
        else:
            file = dataset_class + ' t%04d.npz'%(i)
        np.savez(os.path.join(save, file), pos1=p1, pos2=p2, flow=f)


def creat_dataset_real(data_root, dataset_name, mode):
    save = os.path.join(data_root, 'particle-flow', dataset_name, mode)
    if not os.path.exists(save):
        os.makedirs(save)
    tree = ET.parse(os.path.join(data_root, dataset_name, mode, '_Tracks.xml'))

    root = tree.getroot()
    flow_dict = []
    pos1 = []
    pos2 = []
    flow = []
    for particle in root.findall('.//particle'):
        for i, detection1 in enumerate(particle.findall('detection')):
            if i < len(particle.findall('detection'))-2: 
                t1 = int(detection1.get('t'))
                x1 = float(detection1.get('x'))
                y1 = float(detection1.get('y'))
                detection2 = particle.findall('detection')[i+2]
                x2 = float(detection2.get('x'))
                y2 = float(detection2.get('y'))  
                if t1 >= len(flow_dict):
                    for j in range(t1 - len(flow_dict) + 1):
                        flow_dict.append({})
                        pos1.append([])
                        pos2.append([])
                        flow.append([])
                flow_dict[t1][(x1, y1)] = [round(x2-x1, 3), round(y2-y1, 3)]
                pos1[t1].append([x1, y1])
                pos2[t1].append([x2, y2])
                flow[t1].append([round(x2-x1, 3), round(y2-y1, 3)])

    print("frame num:", len(pos1))
    for i in range(len(pos1)):
        f = np.array(flow[i])
        p1 = np.array(pos1[i])
        p2 = np.array(pos2[i])

        # Sort vertices by y then x
        sort_inds = np.lexsort(p1.T)
        p1 = p1[sort_inds]
        f = f[sort_inds]
        sort_inds = np.lexsort(p2.T)
        p2 = p2[sort_inds]

        file = dataset_name + ' t%03d.npz'%(i)
        np.savez(os.path.join(save, file), pos1=p1, pos2=p2, flow=f)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='data/',) 
args = parser.parse_args()

mode = ['train', 'test']
dataset_name = ['VESICLE', 'MICROTUBULE', 'RECEPTOR']
snr = 7
density = ['high', 'low']
for name in dataset_name:
    for den in density:
        for m in mode:
            data_root = os.path.join(args.root, 'SIMULATED', m)
            if m == 'train':
                inx = [1,2,3]
                for i in inx:
                    creat_dataset(data_root, name, den, snr, i)
            else:
                creat_dataset(data_root, name, den, snr)
        
mode = ['train', 'test']
data_root = os.path.join(args.root, 'REAL')
dataset_name = ['EB1', 'CCR5', 'LYSOSOME']
for name in dataset_name:
    for m in mode:
        creat_dataset_real(data_root, name, m)