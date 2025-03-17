
import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
import numpy as np
import cv2
from tqdm import tqdm

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from src.DensePose.densepose import add_densepose_config
from src.DensePose.densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput


from src.DensePose.densepose.vis.extractor import (
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
)
import pathlib
from multiprocessing import Pool, Queue
import multiprocessing as mp
from functools import partial
from preprocess_data.texture_interface import convert_texture_inpainting

def setup_config(config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(args.opts)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg

def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((i, uv[0, :, :], uv[1, :, :]))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv

def process_output(outputs, file_name, image_path):
    result = {}
    # parse useful information from outputs
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()
            result["pred_densepose"] = extractor(outputs)[0]
    
    # parse iuv from result
    try:
        bbox = result['pred_boxes_XYXY'][0]
    except:
        print("{} not detected !".format(file_name))
        return None, None
    
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    x, y, w, h = [int(v) for v in bbox]
    iuv = parse_iuv(result)
    image = cv2.imread(image_path + file_name)
    img_h, img_w, _ = image.shape
    bg = np.zeros((img_h, img_w, 3))
    bg[y:y + h, x:x + w, :] = iuv
    bg = bg.astype(np.uint8)

    return bg, image
    
if __name__ == "__main__":
    
    config_path = "./src/DensePose/config/densepose_rcnn_R_101_FPN_DL_s1x.yaml"
    model_path = "./preprocess_data/model_final_dl_s1x_101.pkl"
    image_path = "./dataset/fashion/train/"
    save_path = "./dataset/fashion/train_densepose/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Loading config from {config_path}")
    cfg = setup_config(config_path, model_path, argparse.Namespace(opts=[],), [])
    print(f"loading model from {model_path}")
    predictor = DefaultPredictor(cfg)
    image_names = os.listdir(image_path)
    print("Processing images to Densepose...")
    for image_name in tqdm(image_names):
        image = read_image(os.path.join(image_path, image_name), format="BGR")
        with torch.no_grad():
            outputs = predictor(image)["instances"]
        bg, img = process_output(outputs, image_name, image_path)
        tex = convert_texture_inpainting(img, bg)
        print("aaa")
        # cv2.imwrite(save_path + image_name.split('.')[0] + '.png', bg)
    print("saaa")