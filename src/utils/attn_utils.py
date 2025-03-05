import torch
from einops import rearrange
import math
from torchvision.transforms import Resize, InterpolationMode
import numpy as np
import cv2


def process_attnmap(attn_map_list, dst_size=[64, 64], classifier_free_guidance=True):
    clean_attn_map = []
    rsz = Resize(dst_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    for attn_map in attn_map_list:
        if classifier_free_guidance:
            attn_map = attn_map[1]
            attn_map = attn_map[:, :, -1]
        height = int(math.sqrt(attn_map.shape[1]))
        attn_map = rearrange(attn_map, 'b (h w)-> b h w', h=height)
        if not attn_map.shape[1] == dst_size[0]:
            attn_map = rsz(attn_map)
        attn_map = torch.mean(attn_map, dim=0)
        clean_attn_map.append(torch.clone(attn_map))
    attnmap = torch.mean(torch.stack(clean_attn_map, dim=0), dim=0)
    attnmap = attnmap.cpu().numpy()
    # attnmap = attnmap - attnmap.min() / (attnmap.max() - attnmap.min())
    attnmap = attnmap / attnmap.max()
    attnmap[attnmap < 0] = 0
    attnmap = (attnmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attnmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # attnmap_out = torch.stack([attnmap, attnmap, attnmap], dim=0)
    return heatmap