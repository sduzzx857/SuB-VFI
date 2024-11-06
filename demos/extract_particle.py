import xml.etree.ElementTree as ET
import numpy as np
import os
import tifffile as tf

def extract_particle_location():
    save = 'demos/particle-loc'
    xml_path = 'demos/images/_Tracks.xml'
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
    for i in range(len(pos)):
        p = np.array(pos[i])

        # Sort vertices by y then x
        sort_inds = np.lexsort(p.T)
        p = p[sort_inds]

        file = 't%04d.npz'%(i)
        np.savez(os.path.join(save, file), pos=p)

        

extract_particle_location()