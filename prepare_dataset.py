import os
import shutil
import torch 
import argparse
import cv2
from tqdm import tqdm
import numpy as np

from preprocess_data.densepose_interface import setup_config, process_output, DefaultPredictor, read_image
from preprocess_data.texture_interface import convert_texture_inpainting

from multiprocessing import Pool, Queue
import multiprocessing as mp
from functools import partial
from insightface.app import FaceAnalysis

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, save_dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    train_root = save_dir + '/train'
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = save_dir + '/test'
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = []
    train_f = open('./data/train.lst', 'r')
    for lines in train_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines)

    test_images = []
    test_f = open('./data/test.lst', 'r')
    for lines in test_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)

    count_train = 0
    count_test = 0
    for root, _, fnames in sorted(os.walk(dir)):
        if "MEN" not in root:
            continue
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path_names = path.split('/')
                print(path_names)
                path_names[5] = path_names[5].replace('_', '')
                path_names[6] = path_names[6].split('_')[0] + "_" + "".join(path_names[6].split('_')[1:])
                path_names = "".join(path_names[2:])
                if path_names in train_images:
                    shutil.copy(path, os.path.join(train_root, path_names))
                    count_train += 1
                if path_names in test_images:
                    shutil.copy(path, os.path.join(test_root, path_names))
                    count_test += 1
    print("total train images: ", count_train)
    print("total test images: ", count_test)

def prepare_pose(image_path, densepose_save_path, config_path, model_path):
    if not os.path.exists(densepose_save_path):
        os.makedirs(densepose_save_path)
    
    print(f"Loading config from {config_path}")
    cfg = setup_config(config_path, model_path, argparse.Namespace(opts=[],), [])
    print(f"loading model from {model_path}")
    predictor = DefaultPredictor(cfg)
    image_names = os.listdir(image_path)
    print("Processing images to Densepose...")
    for image_name in tqdm(image_names):
        if os.path.exists(densepose_save_path + image_name.split('.')[0] + '.png'):
            continue
        
        # get densepose
        image = read_image(os.path.join(image_path, image_name), format="BGR")
        with torch.no_grad():
            outputs = predictor(image)["instances"]
        bg, _ = process_output(outputs, image_name, image_path)
        if bg is None:
            continue
        cv2.imwrite(densepose_save_path + image_name.split('.')[0] + '.png', bg)
    
    return True

def prepare_texture_mp(image_name):
    densepose_name = image_name.replace("train", "train_densepose").replace("test", "test_densepose")
    densepose_name = os.path.splitext(densepose_name)[0] + '.png'
    texture_name = densepose_name.replace("_densepose", "_texture")
    tex = convert_texture_inpainting(image_name, densepose_name)
    if tex is not None:
        print("Saving texture to ", texture_name)
        cv2.imwrite(texture_name, (tex*255).astype(np.uint8))
    else:
        print(f"Failed to process {densepose_name}")
    return True

def prepare_face(image_path, face_save_path, super_resolution=False):
    face_embedder = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_embedder.prepare(ctx_id=0, det_size=(640, 640))
    
    if not os.path.exists(face_save_path):
        os.makedirs(face_save_path)
    
    image_names = os.listdir(image_path)
    print("Processing images to face...")
    for image_name in tqdm(image_names):
        image = cv2.imread(image_path + image_name)
        face_info = face_embedder.get(image)
        if len(face_info) == 0:
            print("No face detected in :", image_name)
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        if super_resolution:
            # We recommend to do super resolution or bicubic interpolation for small faces, which could slightly improve the overall performance.
            # "But currently our code do not implement it. 
            raise NotImplementedError
        else:
            face_emb = face_info['embedding']
            save_name = face_save_path + os.path.splitext(image_name)[0] + '.npy'
            np.save(save_name, face_emb)

if __name__ == "__main__":
    base_dataset_dir = './dataset/fashion/'
    
    # split the train and test dataset.
    make_dataset('./dataset/fashion/', './dataset/fashion/')
    
    
    # use Densepose to estimate the poses.
    densepose_train_dir = './dataset/fashion/train_densepose/'
    densepose_test_dir = './dataset/fashion/test_densepose/'
    densepose_config_path = "./src/DensePose/config/densepose_rcnn_R_101_FPN_DL_s1x.yaml"
    densepose_model_path = "./pretrained_weights/model_final_844d15.pkl"
    prepare_pose(base_dataset_dir + 'train/', densepose_train_dir, \
         densepose_config_path, densepose_model_path)
    prepare_pose(base_dataset_dir + 'test/', densepose_test_dir, \
        densepose_config_path, densepose_model_path)
    
    
    # parallel process the images to get the texture map.
    all_names = []
    image_names_train = os.listdir(base_dataset_dir + 'train/')
    image_names_test = os.listdir(base_dataset_dir + 'test/')
    all_names.extend(list(base_dataset_dir + 'train/' + name for name in image_names_train))
    all_names.extend(list(base_dataset_dir + 'test/'+ name for name in image_names_test))
    os.makedirs(base_dataset_dir + 'train_texture/', exist_ok=True)
    os.makedirs(base_dataset_dir + 'test_texture/', exist_ok=True)
    
    with Pool(processes=8, maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    prepare_texture_mp,
                ),
                all_names
            ),
            total = len(all_names)
        ):
            pass    
    
    # use arcface to estimate the faces.
    prepare_face(base_dataset_dir + 'train/', base_dataset_dir + 'train_face/')
    prepare_face(base_dataset_dir + 'test/', base_dataset_dir + 'test_face/')