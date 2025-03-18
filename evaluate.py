import os
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
import cv2 
import argparse
from validation.metrics import build_metric
from src.dataset.deepfashion_dataset import FidRealDeepFashion
from insightface.app import FaceAnalysis

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_fid_deepfashion(fid_real_loader, pred_gathered, metric):
    """
        Note that the FID code used in pose transfer is slightly different from the official pytorch FID implementation.
        Details see:
            https://github.com/YanzuoLu/CFLD/issues/43
    """
    gt_out_gathered = []
    print("computing FID, need some time to tranverse all training images.")
    for i, fid_real_imgs in enumerate(tqdm(fid_real_loader)):
        gt_out = metric(fid_real_imgs.to('cuda'))
        gt_out_gathered.append(gt_out.cpu().numpy())

    gt_out_gathered = np.concatenate(gt_out_gathered, axis=0)
    
    mu1 = np.mean(gt_out_gathered, axis=0)
    sigma1 = np.cov(gt_out_gathered, rowvar=False)
    mu2 = np.mean(pred_gathered, axis=0)
    sigma2 = np.cov(pred_gathered, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % 1e-6
        # logger.info(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid_score    

def get_fid_loader(dataset_path='./dataset/fashion/', resolution=256):
    if resolution == 256:
        image_size = [256, 176]
    else:
        image_size = [512, 176*2]
    fid_real_data = FidRealDeepFashion(dataset_path, image_size)
    fid_real_loader = torch.utils.data.DataLoader(
        fid_real_data,
        1, 
        num_workers=2,  
        pin_memory=True
    )

    return fid_real_loader

def evaluate_whole_image(img_path, gt_path, training_path, metric=None, resolution=256):
    image_names = os.listdir(gt_path + '/{}/'.format(resolution))
    
    pred_out_gathered = []
    lpips_gathered = []
    psnr_gathered = []
    ssim_gathered = []
    ssim_256_gathered = []
    
    for idx, image_name in enumerate(tqdm(image_names)):
        # load the gt, generated image as 256, 512 resolution
        gt = pil_to_tensor(Image.open(gt_path + '/{}/'.format(resolution) + image_name)).to(device='cuda', dtype=torch.float32).unsqueeze(0) / 255.
        img = pil_to_tensor(Image.open(img_path + '/{}/'.format(resolution) + image_name)).to(device='cuda', dtype=torch.float32).unsqueeze(0)  / 255.

        pred_out, lpips, psnr, ssim, ssim_256 = metric(gt, img)
        
        pred_out_gathered.append(pred_out.cpu().numpy())
        lpips_gathered.append(lpips.cpu().numpy())
        psnr_gathered.append(psnr.cpu().numpy())
        ssim_gathered.append(ssim.cpu().numpy())
        ssim_256_gathered.append(ssim_256.cpu().numpy())

    pred_out_gathered = np.concatenate(pred_out_gathered, axis=0)
    lpips_gathered = np.concatenate(lpips_gathered, axis=0)
    psnr_gathered = np.concatenate(psnr_gathered, axis=0)
    ssim_gathered = np.concatenate(ssim_gathered, axis=0)
    ssim_256_gathered = np.concatenate(ssim_256_gathered, axis=0)
    
    score_lpips = np.mean(lpips_gathered)
    score_ssim = np.mean(ssim_gathered)
    score_ssim_256 = np.mean(ssim_256_gathered)
    score_psnr = np.mean(psnr_gathered)
    
    # FID computing
    fid_real_loader = get_fid_loader(training_path)
    score_fid = compute_fid_deepfashion(fid_real_loader, pred_out_gathered, metric=metric)
    
    print("Evaluation Results on {} resolution:".format(resolution))
    print("FID: {}".format(score_fid))
    print("LPIPS: {}".format(score_lpips))
    print("SSIM: {}".format(score_ssim))
    print("SSIM_256: {}".format(score_ssim_256))
    print("PSNR: {}".format(score_psnr))
    print("------------------------------------------")

def get_face_emb(face_app, face_image, return_bbox=False):
    face_info = face_app.get(face_image)
    if len(face_info) == 0:
        return None
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    if return_bbox:
        bbox = face_info['bbox'].astype(int)
        bbox[bbox < 0] = 0
        return face_emb, bbox
    return face_emb

def compute_similarity(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    euclidean = np.linalg.norm(a - b)
    return cosine, euclidean

def face_computing(source_image, image, app):
    image = cv2.resize(image, (source_image.shape[1], source_image.shape[0]))
    emb1 = get_face_emb(app, image)
    emb2 = get_face_emb(app, source_image)
    if emb1 is None or emb2 is None:
        return None, None
    cosine, euclidean = compute_similarity(emb2, emb1)
    return cosine, euclidean

def evaluate_face(img_path, gt_path, face_detector):
    image_names = os.listdir(gt_path + '/512/')
    cosine_gathered = []
    eucli_gathered = []
    for idx, image_name in enumerate(tqdm(image_names)):
        gt = cv2.imread(gt_path + '/512/' + image_name)
        img = cv2.imread(img_path + '/512/' + image_name)
        cosine, eucli = face_computing(img, gt, app=face_detector)
        if cosine is not None:
            cosine_gathered.append(cosine)
            eucli_gathered.append(eucli)
    score_cos = np.mean(cosine_gathered)
    score_distance = np.mean(eucli_gathered)
    print("Evaluation Results on faces:")
    print("Face cosine similarity: {}".format(score_cos))
    print("Face euclidean distance : {}".format(score_distance))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="./results/gt/")
    parser.add_argument("--img_path", type=str, default="./results/diff_mask2/")
    parser.add_argument("--training_path", type=str, default="./dataset/fashion/train/")
    args = parser.parse_args()

    # set the metric for whole image 
    metric = build_metric().to('cuda')
    evaluate_whole_image(args.img_path, args.gt_path, args.training_path, metric, resolution=256)
    evaluate_whole_image(args.img_path, args.gt_path, args.training_path, metric, resolution=512)
    
    # set the metric for faces  
    face_detector = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider'])
    face_detector.prepare(ctx_id=0, det_size=(512, 352))
    evaluate_face(args.img_path, args.gt_path, face_detector)