from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
import torch
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_diff_prompt import UNet3DMultiConditionModel
from transformers import CLIPVisionModelWithProjection
from torchvision import transforms
from train import set_ip_adapter
from ip_adapter.resampler import Resampler


class BasePipelineLoader:
    def __init__(self, cfg_path):
        self.cfg = OmegaConf.load(cfg_path)
        self.weight_dtype = torch.float32
    
    def init_model(self, ckpt_dir=None, ckpt_iters=0):
        # initialize model parameters and pipelines, also noise scheduler
        pass 
    
    def forward(self, batch_input):
        pass

class MCLDPipelineLoader(BasePipelineLoader):
    def __init__(self, cfg_path, num_tokens=16, **kwargs):
        super().__init__(cfg_path=cfg_path)
        
        inference_config_path = self.cfg.inference_config
        self.infer_config = OmegaConf.load(inference_config_path)
        sched_kwargs = OmegaConf.to_container(self.infer_config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)
        
        # load the checkpoints
        self.denoising_unet = UNet3DMultiConditionModel.from_pretrained_2d(
            self.cfg.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=self.infer_config.unet_additional_kwargs,
        ).to(dtype=self.weight_dtype, device="cuda")
        self.denoising_unet = set_ip_adapter(self.denoising_unet, num_tokens=num_tokens)

        num_tokens_image_proj = num_tokens
        image_emb_dim_image_proj = 512
        self.image_proj_model =  Resampler(
            dim=1280,
            depth=4, 
            dim_head=64,
            heads=20,
            num_queries=num_tokens_image_proj,
            embedding_dim=image_emb_dim_image_proj,
            output_dim=self.denoising_unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(dtype=self.weight_dtype, device="cuda")

        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_vae_path,
        ).to(device="cuda", dtype=self.weight_dtype)
            
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.image_encoder_path
        ).to(dtype=self.weight_dtype, device="cuda")
            
        self.pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=self.weight_dtype, device="cuda"
        )
               
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=self.weight_dtype, device="cuda")

        self.generator = torch.manual_seed(42)
        self.to_pil = transforms.ToPILImage()
        self.pipe = None
        
    def init_model(self, ckpt_dir=None, ckpt_iters=0, pipe_class=Pose2ImagePipeline):    
        if ckpt_dir is not None:
            self.cfg.denoising_unet_path = ckpt_dir + 'denoising_unet-{}.pth'.format(ckpt_iters)
            self.cfg.reference_unet_path = ckpt_dir + 'reference_unet-{}.pth'.format(ckpt_iters)
            self.cfg.pose_guider_path = ckpt_dir + 'pose_guider-{}.pth'.format(ckpt_iters)
            self.cfg.image_proj_model_path= ckpt_dir + 'image_proj_model-{}.pth'.format(ckpt_iters)
        # load pretrained weights
        self.denoising_unet.load_state_dict(
            torch.load(self.cfg.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        self.reference_unet.load_state_dict(
            torch.load(self.cfg.reference_unet_path, map_location="cpu"),
        )
        self.pose_guider.load_state_dict(
            torch.load(self.cfg.pose_guider_path, map_location="cpu"),
        )
        
        if self.image_proj_model is not None:
            self.image_proj_model.load_state_dict(
                torch.load(self.cfg.image_proj_model_path, map_location="cpu"),
            )                 
        
        print("all model loaded form {} ! ".format(ckpt_dir))
        
        pipe = pipe_class(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler,
            image_proj_model=self.image_proj_model,
        )
        
        self.pipe = pipe.to("cuda", dtype=self.weight_dtype)

    def load_image(self, batch_input):

        ref_image_pil = batch_input['ref_img']
        pose_image_pil = batch_input['tgt_pose']
        input_image_pil = batch_input['img_input']
        clip_image_pil = batch_input['clip_origin']
        clip_image_pil2 = batch_input['clip_origin2']
        ref_image_pil = ref_image_pil if self.cfg.data.texture_unet_condition else input_image_pil
        face_emb = batch_input['face_emb']
            
        return ref_image_pil.to(self.vae.device), pose_image_pil.to(self.vae.device), clip_image_pil.to(self.vae.device), clip_image_pil2.to(self.vae.device), face_emb.to(self.vae.device)

    def forward(self, ref_image_pil, pose_image_pil, clip_image_pil, width=512, height=512, **kwargs):
        # pdb.set_trace()
        image_ = self.pipe(
            ref_image_pil,
            pose_image_pil,
            width,
            height,
            20,
            3.5,
            generator=self.generator,
            clip_image=clip_image_pil,
            **kwargs,
        )
        image = image_.images[:, :, 0, :, :]
        return image