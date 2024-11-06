import torch

def get_flow_map(h,w, pos, particle_flow_prediction, n):
    # pos1: B*N*2, flow: B*N*2, n: B*1
    B = len(n)
    flow_map_2 = torch.zeros((B, int(h/2),int(w/2), 2)).cuda()
    flow_map_4 = torch.zeros((B, int(h/4),int(w/4), 2)).cuda()
    flow_map_8 = torch.zeros((B, int(h/8),int(w/8), 2)).cuda()
    for k in range(B):

        det = pos[k, :int(n[k])]
        pf = particle_flow_prediction[k, :int(n[k])]
        x_coords, y_coords = det[:, 0], det[:, 1]
        mask_x_min = x_coords >= 0
        mask_x_max = x_coords <= w-1
        mask_y_min = y_coords >= 0
        mask_y_max = y_coords <= h-1
        mask = mask_y_min & mask_y_max & mask_x_min & mask_x_max
        # Ensure the coordinates are within the image boundaries
        x_coords = x_coords[mask]
        y_coords = y_coords[mask] 
        pf = pf[mask]
        # Set the pixels at the given coordinates to 1
        flow_map_2[k, (y_coords/2).long(), (x_coords/2).long()] = pf/2
        flow_map_4[k, (y_coords/4).long(), (x_coords/4).long()] = pf/4
        flow_map_8[k, (y_coords/8).long(), (x_coords/8).long()] = pf/8
 
    return [flow_map_2.permute(0,3,1,2), flow_map_4.permute(0,3,1,2), flow_map_8.permute(0,3,1,2)]


def get_feature_map(h, w, pos, pos_feature, n):
    B, _, C = pos_feature.shape
    feature_maps2 = torch.zeros((B, int(h/2),int(w/2), C)).cuda()
    feature_maps4 = torch.zeros((B, int(h/4),int(w/4), C)).cuda()
    feature_maps8 = torch.zeros((B, int(h/8),int(w/8), C)).cuda()
    
    for k in range(B):
        det = pos[k, :int(n[k])]
        pf = pos_feature[k, :int(n[k])]
        x_coords, y_coords = det[:, 0], det[:, 1]
        mask_x_min = x_coords >= 0
        mask_x_max = x_coords <= w-1
        mask_y_min = y_coords >= 0
        mask_y_max = y_coords <= h-1
        mask = mask_y_min & mask_y_max & mask_x_min & mask_x_max
        # Ensure the coordinates are within the image boundaries
        x_coords = x_coords[mask]
        y_coords = y_coords[mask] 
        pf = pf[mask]
        
        feature_maps2[k, (y_coords/2).long(), (x_coords/2).long()] = pf
        feature_maps4[k, (y_coords/4).long(), (x_coords/4).long()] = pf
        feature_maps8[k, (y_coords/8).long(), (x_coords/8).long()] = pf
    
    feature_maps2 = feature_maps2.permute(0, 3, 1, 2) # B*C*h*w
    feature_maps4 = feature_maps4.permute(0, 3, 1, 2) # B*C*h*w
    feature_maps8 = feature_maps8.permute(0, 3, 1, 2) # B*C*h*w
    return [feature_maps2, feature_maps4, feature_maps8]
