import numpy as np
import cv2
import os
from UVTextureConverter import Normal2Atlas, Atlas2Normal
from PIL import Image
from multiprocessing import Pool, Queue
import multiprocessing as mp
from functools import partial
import pathlib

part_ids = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 10, 10])

bg = cv2.imread('./preprocess_data/texture_bg_200.png', 0)
bg_ = []
for j in range(4):
    for i in range(6):
        tmp = bg[i*200: (i+1) * 200, j*200:(j+1)*200]
        bg_.append(tmp)
bgs = np.stack(bg_, axis=0)

def get_texture(im,IUV, bgs, solution=200):
    #
    #inputs:
    #   solution is the size of generated texture, in notebook provided by facebookresearch the solution is 200
    #   If use lager solution, the texture will be sparser and smaller solution result in denser texture.
    #   im is original image
    #   IUV is densepose result of im
    #output:
    #   TextureIm, the 24 part texture of im according to IUV
    
    solution_float = float(solution) - 1

    U = IUV[:,:,1]
    V = IUV[:,:,2]
    parts = list()
    masks = list()
    full_parts = list()
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        actual_part_ = np.zeros((solution, solution, 3))
        cur_bg = bgs[PartInd - 1]
        x,y = np.where(IUV[:,:,0]==PartInd)
        if len(x) == 0:
            parts.append(actual_part_)
            full_parts.append(actual_part_)
            masks.append(actual_part_)
            continue

        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        tex_map_coords = ((255-v_current_points)*solution_float/255.).astype(int),(u_current_points*solution_float/255.).astype(int)
        for c in range(3):
            actual_part_[tex_map_coords[0],tex_map_coords[1], c] = im[x,y,c]

        actual_part = actual_part_
        valid_mask = np.array((actual_part_.sum(2) != 0) * 1, dtype='uint8')
        radius_increase = 10
        kernel = np.ones((radius_increase, radius_increase), np.uint8)
        dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
        region_to_fill = dilated_mask - valid_mask
        # invalid_region = 1 - valid_mask
        invalid_region = ((cur_bg / 255 - valid_mask) > 0).astype(np.uint8)
        # invalid_region = np.repeat(invalid_region[:, np.newaxis, :], 3, axis=1)
        actual_part_max = actual_part.max()
        actual_part_min = actual_part.min()
        actual_part_uint = np.array((actual_part - actual_part_min) / (actual_part_max - actual_part_min) * 255,
                                    dtype='uint8')
        actual_part_uint = cv2.inpaint(actual_part_uint, invalid_region, 3, cv2.INPAINT_TELEA)
        actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
        # only use dilated part
        mask = np.repeat(dilated_mask[:, :, np.newaxis], 3, axis=2)
        actual_part_out = (mask * actual_part).astype(np.uint8)

        mask_out = np.repeat((dilated_mask* cur_bg)[:, :, np.newaxis] , 3, axis=2)
        masks.append(mask_out)
        parts.append(actual_part_out)
        full_parts.append(actual_part)

    TextureIm  = np.zeros([solution*6,solution*4,3])
    TextureInd  = np.zeros([solution*6,solution*4,3])
    TextureIm_full  = np.zeros([solution*6,solution*4,3])

    for i in range(4):
        for j in range(6):
            TextureIm[(solution*j):(solution*j+solution) , (solution*i):(solution*i+solution) ,: ] = parts[i*6+j]
    
    for i in range(4):
        for j in range(6):
            TextureIm_full[(solution*j):(solution*j+solution) , (solution*i):(solution*i+solution) ,: ] = full_parts[i*6+j]
    
    for i in range(4):
        for j in range(6):
            TextureInd[(solution*j):(solution*j+solution) , (solution*i):(solution*i+solution) ,: ] = masks[i*6+j]

    return TextureIm, TextureIm_full, TextureInd


def convert_texture_inpainting(img_path, iuv_path):
    iuv = cv2.imread(iuv_path)
        # im = np.zeros(iuv.shape)
    im = cv2.imread(img_path)
    
    try:
        texim, texim_full, texim_ind, = get_texture(im, iuv, bgs)
    except:
        return None
    texim = texim.astype(np.uint8)
    texim_full = texim_full.astype(np.uint8)
    texim_ind = texim_ind.astype(np.uint8)
    
    tex_input = texim.transpose(1, 0, 2)

    atlas_tex_stack = Atlas2Normal.split_atlas_tex(tex_input)
    converter = Atlas2Normal(atlas_size=200, normal_size=512)
    normal_tex_2 = converter.convert(atlas_tex_stack)
    
    return normal_tex_2