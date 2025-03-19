"""
@origin author: Yanzuo Lu
"""

import glob
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import json 
import tqdm
import numpy as np
import pdb

logger = logging.getLogger()


def load_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)
    

class DeepFashion(Dataset):
    def __init__(self, 
                 root_dir, 
                 image_size=512, 
                 do_normalize=True,
                 texture_clip_condition=False,  # if we use texture as the clip condition in refnet
                 texture_unet_condition=False,  # if we use texture as the unet conditon in refnet 
                 if_train=True,
                 double_clip = False,
                 use_dino = False,
                 ):
        super().__init__()

        self.image_size = image_size
        # root_dir = os.path.join(root_dir, "DeepFashion")
        data_dir = root_dir
        
        self.double_clip = double_clip
        
        if if_train:
            pair_name = './data/train.csv'
            self.img_dir = data_dir + 'train/'
            self.pose_path = data_dir + 'train_densepose/'
        else:
            pair_name = './data/test.csv'
            self.img_dir = data_dir + 'test/'
            self.pose_path = data_dir + 'test_densepose/'
        pairs = os.path.join('', pair_name)
        pairs = pd.read_csv(pairs)
        self.img_items = self.process_dir(data_dir, pairs)

        self.texture_clip_condition = texture_clip_condition
        self.texture_unet_condition = texture_unet_condition

        self.clip_image_processor = CLIPImageProcessor()

        transform = [transforms.ToTensor()]
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.base_transform = transforms.ToTensor()

        self.test_512 = transforms.Compose([
            transforms.Resize([512, 176 * 2], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])

        self.test_256 = transforms.Compose([
            transforms.Resize([256, 176], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])
        
        self.use_dino = use_dino
        self.dino_preprocess = transforms.Compose([
            transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_dir(self, root_dir, csv_file):
        data = []
        for i in range(len(csv_file)):
            data.append((csv_file.iloc[i]["from"],  csv_file.iloc[i]["to"]))
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        print("this class is deprecated now.")
        raise NotImplementedError

            
class DeepFashionBaseline(DeepFashion):
    def __init__(self, 
                 root_dir, 
                 image_size=512, 
                 do_normalize=True,
                 texture_clip_condition=False,  # if we use texture as the clip condition in refnet
                 texture_unet_condition=False,  # if we use texture as the unet conditon in refnet 
                 if_train=True,
                 pose_cond_type = 'dwpose',
                 clip_cond_type = 'whole_image',
                 double_clip = False,
                 use_face_emb = False,
                 use_dino = False,
                 ):
        super().__init__(root_dir=root_dir,
                         image_size=image_size,
                         do_normalize=do_normalize,
                         texture_clip_condition=texture_clip_condition,
                         texture_unet_condition=texture_unet_condition,
                         if_train=if_train,
                         double_clip=double_clip,
                         use_dino=use_dino)
        self.pose_cond_type = pose_cond_type
        self.clip_cond_type = clip_cond_type
        self.use_face_emb = use_face_emb

    
    def __getitem__(self, index):
        img_path_from, img_path_to = self.img_items[index]
        img_gt_ = Image.open(self.img_dir + img_path_to)        
        img_input = Image.open(self.img_dir + img_path_from).resize((512, 512))

        # change the postfix 
        image_path_from = os.path.splitext(image_path_from) + '.png'
        image_path_to = os.path.splitext(image_path_to) + '.png'

        if self.pose_cond_type == 'dwpose':
            raise NotImplementedError
        elif self.pose_cond_type == 'densepose':
            pose_path = self.pose_path + image_path_to
        else:
            raise NotImplementedError

        texture_path = self.densepose_dir.replace('densepose/', 'texture/') + image_path_from
        texture_img = Image.open(texture_path).resize((512, 512))

        pose_img = Image.open(pose_path)
        
        ref_face_emb = None
        ref_clip_input = texture_img if self.texture_clip_condition else img_input
        ref_unet_input = texture_img if self.texture_unet_condition else img_input

        img_gt = self.transform(img_gt_.resize((512, 512)))
        # add for inpput parts 
        
        ref_img = self.transform(ref_unet_input)
        img_input_ = self.transform(img_input.resize((512, 512)))
        pose_image = self.base_transform(pose_img.resize((512, 512)))
        clip_origin_image = self.base_transform(ref_clip_input)
        if self.use_dino:
            clip_image = self.dino_preprocess(ref_clip_input)
        else:
            clip_image = self.clip_image_processor(images=ref_clip_input, return_tensors="pt").pixel_values[0]

        gt_256 = self.test_256(img_gt_)
        gt_512 = self.test_512(img_gt_)
        
        # for easily locate the image location
        combined_image_name = img_path_from[:-4] + '_2_' + img_path_to[:-4]
        
        sample = dict(
            image_name=img_path_to,
            ref_img=ref_img,
            img=img_gt,
            img_input=img_input_,
            clip_images=clip_image,
            clip_origin=clip_origin_image,
            tgt_pose=pose_image, 
            gt_256=gt_256,
            gt_512=gt_512,   
            save_image_name = combined_image_name,
        )

        if self.use_face_emb:
            face_emb_path = self.densepose_dir.replace('densepose/', 'face/') + image_path_from.replace('png', 'npy')
            if os.path.exists(face_emb_path):
                ref_face_emb = np.load(face_emb_path)
                ref_face_emb = np.expand_dims(ref_face_emb, axis=0)
            else:
                ref_face_emb = np.zeros((1, 512), dtype=float)
            sample["face_emb"] = ref_face_emb
            
        if self.double_clip:
            double_clip = img_input if self.texture_clip_condition else texture_img
            if self.use_dino:
                clip_image2 = self.dino_preprocess(double_clip)
            else:
                clip_image2 = self.clip_image_processor(images=double_clip, return_tensors="pt").pixel_values[0]
            sample["clip_images2"] = clip_image2
            origin_clip2 = self.base_transform(double_clip)
            sample["clip_origin2"] = origin_clip2
        
        return sample

def get_deepfashion_dataset(cfg, **kargs):

    train_dataset = DeepFashionBaseline(
        root_dir= cfg.data.root_dir,
        image_size=cfg.data.train_width,
        texture_clip_condition=cfg.data.texture_clip_condition,
        texture_unet_condition=cfg.data.texture_unet_condition,
        if_train=True,
        pose_cond_type=cfg.data.pose_type,
        clip_cond_type=cfg.data.clip_type,
        **kargs,
    )

    val_dataset = DeepFashionBaseline(
        root_dir= cfg.data.root_dir,
        image_size=cfg.data.train_width,
        texture_clip_condition=cfg.data.texture_clip_condition,
        texture_unet_condition=cfg.data.texture_unet_condition,
        do_normalize=False,
        if_train=False,
        pose_cond_type=cfg.data.pose_type,
        clip_cond_type=cfg.data.clip_type,
        **kargs,
    )
    return train_dataset, val_dataset

class FidRealDeepFashion(Dataset):
    def __init__(self, root_dir, test_img_size):
        super().__init__()
        # root_dir = os.path.join(root_dir, "DeepFashion")
        train_dir = os.path.join(root_dir, "train")
        self.img_items = self.process_dir(train_dir)

        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def process_dir(self, root_dir):
        data = []
        img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        for img_path in img_paths:
            data.append(img_path)
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path = self.img_items[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform_test(img)