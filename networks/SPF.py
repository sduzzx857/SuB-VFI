import torch
import torch.nn as nn
from networks.gmsf import GMSF
from networks.AMT_L import Model
from networks.particle_utils import get_flow_map, get_feature_map

class SPF(nn.Module):
    def __init__(self, gmsf_pretrained='', syn_pretrained=''):
        super(SPF, self).__init__()
        
        self.particle_flow = GMSF('DGCNN', pretrained=gmsf_pretrained)
        self.conv = nn.Conv1d(64, 16, 1)
        self.synnet = Model(pretrained=syn_pretrained)

    def forward(self, pos0, pos1, n0, n1, img0, img1, embt, eval=False, **kwargs):
        # B x N x 2
        pos0_center = torch.mean(pos0, 1, keepdim=True)
        pos0 -= pos0_center
        pos1 -= pos0_center
        _,_,h,w = img0.shape

        pos_scale = 8
        pos0 = pos0 / pos_scale
        pos1 = pos1 / pos_scale

        particle_flow_prediction0, particle_flow_prediction1, pos_feature0, pos_feature1 = self.particle_flow(pos0, pos1)
        pos_feature0 = self.conv(pos_feature0)
        pos_feature1 = self.conv(pos_feature1)
        particle_flow_prediction0 = particle_flow_prediction0.permute(0,2,1)
        particle_flow_prediction1 = particle_flow_prediction1.permute(0,2,1)
        pos_feature0 = pos_feature0.permute(0,2,1)
        pos_feature1 = pos_feature1.permute(0,2,1)

        pos0 = pos0 * pos_scale
        pos1 = pos1 * pos_scale
        pos0 += pos0_center
        pos1 += pos0_center
        particle_flow_prediction0 = particle_flow_prediction0 * pos_scale
        particle_flow_prediction1 = particle_flow_prediction1 * pos_scale
        pos0t = pos0 + particle_flow_prediction0*0.5
        pos1t = pos1 + particle_flow_prediction1*0.5

        flow_map_0 = get_flow_map(h,w,pos0t,-particle_flow_prediction0*0.5,n0)
        flow_map_1 = get_flow_map(h,w,pos1t,-particle_flow_prediction1*0.5,n1)
        
        feature_map_0 = get_feature_map(h,w,pos0t,pos_feature0,n0)
        feature_map_1 = get_feature_map(h,w,pos1t,pos_feature1,n1)
        
        output = self.synnet(img0, img1, feature_map_0, feature_map_1, flow_map_0, flow_map_1, embt, eval=eval)

        return output