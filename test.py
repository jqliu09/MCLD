import torch
import torch.nn.functional as F
import os 
import argparse
from validation.pipeline_loader import MCLDPipelineLoader
from src.dataset.deepfashion_dataset import  get_deepfashion_dataset
from src.pipelines.pipeline_pose2img_mcld import Pose2ImageIpMultiPipeline
from omegaconf import OmegaConf
import tqdm
from torchvision.transforms import Resize
from torchvision import transforms

to_pil = transforms.ToPILImage()
resz = Resize((512, 512))

def inference_func(cur_batch, pipeloader):
    ref, pos, clip, clip2, face_emb = pipeloader.load_image(cur_batch)
    face_emb = face_emb.to(dtype=pipeloader.vae.dtype)
    out_image = pipeloader.forward(ref, pos, clip * 255, clip_image2 = clip2 * 255, face_emb=face_emb)
    return out_image

def postprocess_image(generated_image, img_shape):
    sampling_imgs = F.interpolate(generated_image, img_shape, mode="bicubic", antialias=True)
    sampling_imgs = sampling_imgs.float() * 255.0
    sampling_imgs = sampling_imgs.clamp(0, 255).to(dtype=torch.uint8)
    sampling_imgs = sampling_imgs.to(torch.float32) / 255.
    
    return sampling_imgs

def test(pipeloader, val_dataloader, infer_func, save_dir='./results/', exp_name='mcld'):
    for step, batch in enumerate(tqdm.tqdm(val_dataloader)):
        images = infer_func(batch, pipeloader)
        img_256 = postprocess_image(images, img_shape=(256, 176))
        img_512 = postprocess_image(images, img_shape=(512, 352))
        batch_size = img_512.shape[0]
        for idx in range(batch_size):
            img_256_save = img_256[idx]
            img_512_save = img_512[idx]
            
            # make save dir and names 
            save_name = batch['save_image_name'][idx]
            save_path = save_dir + exp_name 
            if not os.path.exists(save_path):
                os.makedirs(save_path + '/256')
                os.makedirs(save_path + '/512')
            save_name_256 = save_path + '/256/' + save_name + '.png'
            save_name_512 = save_path + '/512/' + save_name + '.png'
            
            # saved here
            img_256_pil = to_pil(img_256_save)
            img_256_pil.save(save_name_256)
            img_512_pil = to_pil(img_512_save)
            img_512_pil.save(save_name_512)
            
    print("all test images are generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/")
    parser.add_argument("--config_path", type=str, default="./configs/train/train.yaml")
    args = parser.parse_args()
    
    dataset_kwargs = {}
    
    # load pipeline
    cur_pipeloader = MCLDPipelineLoader(args.config_path, num_tokens=16, image_proj_model=True)
    cur_pipeloader.init_model(args.ckpt_dir, pipe_class=Pose2ImageIpMultiPipeline)
    infer_func = inference_func
    dataset_kwargs['double_clip'] = True
    dataset_kwargs['use_face_emb'] = True
    
    cfg = OmegaConf.load(args.config_path)
    _, val_dataset = get_deepfashion_dataset(cfg, **dataset_kwargs)
    
    val_dataloader = torch.utils.data.DataLoader(
       val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, drop_last=False
    )
    
    test(cur_pipeloader, val_dataloader, infer_func=infer_func, save_dir=args.save_path, exp_name=cfg.exp_name)