import numpy as np
import cv2
import os
import tqdm
from UVTextureConverter import Normal2Atlas, Atlas2Normal
from PIL import Image
import random
import multiprocessing


part_ids = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 10, 10])


def TransferTexture(texture_image,im,IUV, inv=False):

    TextureIm = np.zeros([24, 200, 200, 3])
    for i in range(4):
        for j in range(6):
            if inv:
                TextureIm[(6 * i + j), :, :, :] = texture_image[(200 * i):(200 * i + 200), (200 * j):(200 * j + 200), :]
            else:
                TextureIm[(6 * i + j), :, :, :] = texture_image[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), :]

    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        #####
        R = tex[:,:,0]
        G = tex[:,:,1]
        B = tex[:,:,2]
        ###############
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        ##  Get the RGB values from the texture images.
        R_im[IUV[:,:,0]==PartInd] = r_current_points
        G_im[IUV[:,:,0]==PartInd] = g_current_points
        B_im[IUV[:,:,0]==PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:,:,np.newaxis],G_im[:,:,np.newaxis],R_im[:,:,np.newaxis]), axis =2 ).astype(np.uint8)
    BG_MASK = generated_image==0
    generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image, BG_MASK


def get_texture(im,IUV,solution=200):
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
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        actual_part = np.zeros((solution, solution, 3))
        x,y = np.where(IUV[:,:,0]==PartInd)
        if len(x) == 0:
            parts.append(actual_part)
            continue

        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        tex_map_coords = ((255-v_current_points)*solution_float/255.).astype(int),(u_current_points*solution_float/255.).astype(int)
        for c in range(3):
            actual_part[tex_map_coords[0],tex_map_coords[1], c] = im[x,y,c]

        valid_mask = np.array((actual_part.sum(2) != 0) * 1, dtype='uint8')
        radius_increase = 10
        kernel = np.ones((radius_increase, radius_increase), np.uint8)
        dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
        region_to_fill = dilated_mask - valid_mask
        invalid_region = 1 - valid_mask
        # invalid_region = np.repeat(invalid_region[:, np.newaxis, :], 3, axis=1)
        actual_part_max = actual_part.max()
        actual_part_min = actual_part.min()
        actual_part_uint = np.array((actual_part - actual_part_min) / (actual_part_max - actual_part_min) * 255,
                                    dtype='uint8')
        actual_part_uint = cv2.inpaint(actual_part_uint, invalid_region, 3, cv2.INPAINT_TELEA)
        actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
        # only use dilated part
        mask = np.repeat(dilated_mask[:, :, np.newaxis], 3, axis=2)
        actual_part = (mask * actual_part).astype(np.uint8)

        parts.append(actual_part)

    TextureIm  = np.zeros([solution*6,solution*4,3])

    for i in range(4):
        for j in range(6):
            TextureIm[(solution*j):(solution*j+solution) , (solution*i):(solution*i+solution) ,: ] = parts[i*6+j]

    return TextureIm


# pose_images_path = "E:/dataset/SHHQ/densepose/iuv/"
# pose_image_name = os.listdir(pose_images_path)
# image_names = os.listdir("E:/dataset/SHHQ/shhq/")
# pose_image_name.sort()
# image_names.sort()


def multi_texture(idx, save_dir="E:/dataset/SHHQ/texture/"):
    image_name = pose_image_name[idx]
    iuv = cv2.imread(pose_images_path + image_name)
    im = cv2.imread("E:/dataset/SHHQ/shhq/" + image_names[idx])
    texim = get_texture(im, iuv)
    cv2.imwrite(save_dir + image_name, texim)
    print("{}  done !".format(idx))
    return 1


def multi_atlas(idx, save_dir="E:/dataset/SHHQ/atlas/"):
    image_name = pose_image_name[idx]
    texture =  np.array(Image.open("E:/dataset/SHHQ/texture/" + image_name).convert('RGB')).transpose(1, 0, 2)
    atlas_tex_stack = Atlas2Normal.split_atlas_tex(texture)
    converter = Atlas2Normal(atlas_size=200, normal_size=512)
    normal_tex = converter.convert(atlas_tex_stack)
    normal_tex = cv2.cvtColor(normal_tex.astype('float32'), cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_dir + image_name, normal_tex*255)
    return 1


if __name__ == "__main__":

    # print(os.cpu_count())
    # pool = multiprocessing.Pool(os.cpu_count() - 1)
    # pool.map(multi_atlas, range(40000))

    pose_images_path = "./pose_process/2/"
    # # pose_images_path = "E:/dataset/SHHQ/densepose/iuv/"
    # save_dir = './pose_process/interpolate_texture_hd/'
    # # save_dir = "E:/dataset/SHHQ/texture/"
    texture_image = cv2.imread("./pose_process/interpolate_texture_hd/image_000001.png")[:,:,::-1]/255.
    pose_image_name = os.listdir(pose_images_path)
    # # img_pth = "E:/dataset/SHHQ/shhq/"
    # img_pth = "./pose_process/image_hd/"
    # image_names = os.listdir(img_pth)
    # pose_image_name.sort()
    # image_names.sort()
    # # get densepose texturemap
    # for idx, image_name in enumerate(tqdm.tqdm(pose_image_name)):
    #     iuv = cv2.imread(pose_images_path + image_name)
    #     # im = np.zeros(iuv.shape)
    #     current_name = image_names[idx]
    #     im = cv2.imread(img_pth + current_name)
    #     texim = get_texture(im, iuv).astype(np.uint8)
    #     cv2.imwrite(save_dir + image_name, texim)
    # #     generated = TransferTexture(texim, im, iuv)
    #     print("aaa")

    # direct render in densepose
    images_video = []
    for image_name in pose_image_name:
        iuv = cv2.imread(pose_images_path + image_name)
        im = np.ones(iuv.shape) * 255
        generated, bg = TransferTexture(texture_image, im, iuv)
        cc = (iuv > 0) * (bg > 0)
        # generated[(iuv > 0) * (generated == 0)] = 100
        images_video.append(generated.copy())

    # # generate_new_images
    # im = np.array(Image.open("./pose_process/interpolate_texture_hd/image_000001.png").convert('RGB')).transpose(1, 0, 2)
    # atlas_tex_stack = Atlas2Normal.split_atlas_tex(im)
    # converter = Atlas2Normal(atlas_size=200, normal_size=512)
    # normal_tex = converter.convert(atlas_tex_stack)
    # normal_tex = cv2.cvtColor(normal_tex.astype('float32'), cv2.COLOR_BGR2RGB)
    # print("aaaa")

    # # statistical
    # ratio_all = []
    # for idx in range(1, 11):
    #     ratio_single = []
    #     texture_image_ = cv2.imread("./pose_process/interpo_texture/image_0000{}.png".format(str(idx).zfill(2)))[:, :, ::-1] / 255.
    #     random.shuffle(pose_image_name)
    #     for image_name in tqdm.tqdm(pose_image_name[0:2000]):
    #         iuv = cv2.imread(pose_images_path + image_name)
    #         im = np.zeros(iuv.shape)
    #         generated = TransferTexture(texture_image_ , im, iuv)
    #         # generated[(iuv > 0) * (generated == 0)] = 100
    #         ratio = np.sum(np.max(generated, axis=2) > 0) / np.sum(np.max(iuv, axis=2) > 0)
    #         ratio_single.append(ratio.copy())
    #         ratio_all.append(ratio.copy())
    #     ratio_np = np.array(ratio_single)
    #     print(np.mean(ratio_np), np.var(ratio_np), np.sqrt(np.var(ratio_np)))
    # ratio_all_np = np.array(ratio_all)
    # print(np.mean(ratio_all_np), np.var(ratio_all_np), np.sqrt(np.var(ratio_all_np)))


