from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataloader.particle import Particle_Flow
from glob import glob

@torch.no_grad()
def validate_things(stage,
                    model,
                    root,dataset,snr,density,npoints
                    ):
    """ Peform validation using the Things (test) split """

    print(dataset+' '+density)

    model.eval()
    if stage == 'particle':
        val_dataset = Particle_Flow(root=root, dataset_name=dataset, snr=snr, density=density, npoints=npoints, train=False)

    print('Number of validation image pairs: %d' % len(val_dataset))
    results = {}
    metrics_3d = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}

    import timeit
    start = timeit.default_timer()
    for val_id in range(len(val_dataset)):
        data_dict = val_dataset[val_id]
        pcs = data_dict['pcs'].unsqueeze(0) # 8192*6
        flow_3d = data_dict['flow_3d'].unsqueeze(0).cuda()
        pc1 = pcs[:,:,0:2].cuda()
        pc2 = pcs[:,:,2:4].cuda()

        results_dict_point = model(pc0 = pc1, pc1 = pc2)
        flow_3d_pred = results_dict_point['flow_preds'][-1]

        if flow_3d[0].shape[0] > 3:
            flow_3d_mask = flow_3d[0][3] > 0
            flow_3d_target = flow_3d[0][:3]
        else:
            flow_3d_mask = torch.ones(flow_3d[0].shape[1], dtype=torch.int64).cuda()
            flow_3d_target = flow_3d[0][:3]

        # test all points including occlusion
        flow_3d_mask = torch.ones(flow_3d[0].shape[1], dtype=torch.int64).cuda()
        flow_3d_target = flow_3d[0][:3]

        epe3d_map = torch.sqrt(torch.sum((flow_3d_pred[0] - flow_3d_target) ** 2, dim=0))

        # evaluate
        flow_3d_mask = torch.logical_and(flow_3d_mask, torch.logical_not(torch.isnan(epe3d_map)))
        metrics_3d['counts'] += epe3d_map[flow_3d_mask].shape[0]
        metrics_3d['EPE3d'] += epe3d_map[flow_3d_mask].sum().item()
        metrics_3d['5cm'] += torch.count_nonzero((epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.05)).item()
        metrics_3d['10cm'] += torch.count_nonzero((epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.1)).item()
        metrics_3d['outlier'] += torch.count_nonzero((epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] > 0.1)).item()
    
    stop = timeit.default_timer()
    print('in-loop Time: ', (stop - start)/len(val_dataset))  

    print('#### 2D Metrics ####')
    results['EPE'] = metrics_3d['EPE3d'] / metrics_3d['counts']
    results['5cm'] = metrics_3d['5cm'] / metrics_3d['counts'] * 100.0
    results['10cm'] = metrics_3d['10cm'] / metrics_3d['counts'] * 100.0
    results['outlier'] = metrics_3d['outlier'] / metrics_3d['counts'] * 100.0
    print("Validation EPE: %.4f, 5cm: %.4f, 10cm: %.4f, outlier: %.4f" % (results['EPE'], results['5cm'], results['10cm'], results['outlier']))

    return results
