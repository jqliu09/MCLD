import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
import pdb

from src.dataset.deepfashion_dataset import get_deepfashion_dataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_diff_prompt import UNet3DMultiConditionModel
from src.pipelines.pipeline_pose2img_mcld import Pose2ImageIpMultiPipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything

from ip_adapter.attention_processors import AttnProcessor2_0 as AttnProcessor, IPAttnProcessor2_0 as IPAttnProcessor
from ip_adapter.resampler import Resampler

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def generate_clip_random_noise(bs, device, dtype, scale_factor=0.05):
    values = []
    std=np.load('./ip_adapter/clip_std.npy')
    for _ in range(bs):
        tmp = torch.normal(mean=0, std=torch.tensor(std))
        tmp = tmp.to(device=device, dtype=dtype)
        values.append(tmp)
    out = torch.stack(values, dim=0)
    out = out.unsqueeze(1) * scale_factor
    return out

class InstanceNet(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DMultiConditionModel,
        pose_guider: PoseGuider,
        image_proj_model: Resampler,
        reference_control_writer,
        reference_control_reader,
        device="cuda"
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.image_proj_model = image_proj_model
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.device = device

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        face_pretrained_embeds,
        clip_image_embeds,
        clip_image_embeds2,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device=self.device)
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        # print(pose_fea.shape)
        # print("aa", ref_image_latents.shape, noisy_latents.shape)
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        
        face_tokens = self.image_proj_model(face_pretrained_embeds)
        
        # will clip be too weak, if in this dimension.
        down_encoder_hidden_states = torch.cat([clip_image_embeds, face_tokens], dim=1)
        mid_encoder_hidden_states = torch.cat([clip_image_embeds, clip_image_embeds2, face_tokens], dim=1)
        up_encoder_hidden_states = torch.cat([clip_image_embeds2, face_tokens], dim=1)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            down_encoder_hidden_states=down_encoder_hidden_states,
            mid_encoder_hidden_states=mid_encoder_hidden_states,
            up_encoder_hidden_states=up_encoder_hidden_states,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def set_ip_adapter(unet, model_ckpt=None, num_tokens=16, scale=0.5):
    attn_procs = {}
    for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim, 
                                                   scale=scale,
                                                   num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
    unet.set_attn_processor(attn_procs)
    
    return unet

def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    data_config,
    net_name,
    num_iters,
    val_dataloader,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider
    image_proj_model = ori_net.image_proj_model

    texture_clip_condition=data_config.texture_clip_condition
    texture_unet_condition=data_config.texture_unet_condition
    width=data_config.train_width
    height=data_config.train_height
    bs = data_config.val_bs

    generator = torch.manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    pipe = Pose2ImageIpMultiPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        image_proj_model=image_proj_model,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    batch = next(iter(val_dataloader))

    to_pil = transforms.ToPILImage()

    h, w = height, width
    canvas = Image.new("RGB", (w * 5 , h * bs), "white")
    for jdx in range(bs):
        
        ref_image_pil = to_pil(batch['ref_img'][jdx])
        pose_image_pil = to_pil(batch['tgt_pose'][jdx])
        gt_image_pil = to_pil(batch['img'][jdx])
        input_image_pil = to_pil(batch['img_input'][jdx])
        face_embs = batch['face_emb'][jdx].to(vae.device, dtype=vae.dtype)
        face_embs = face_embs.unsqueeze(0)

        
        image_out = Image.new('RGB', (512 * 3, 512))
        image_out.paste(ref_image_pil, (0, 0))
        image_out.paste(pose_image_pil, (512, 0))
        image_out.paste(input_image_pil, (512 * 2, 0))
        image_out.save("test_1.png")
        
        clip_image_pil = to_pil(batch['clip_origin'][jdx])
        clip_image_pil2 = to_pil(batch['clip_origin2'][jdx])
        ref_image_pil = ref_image_pil if texture_unet_condition else input_image_pil

        image = pipe(
            ref_image_pil,
            pose_image_pil,
            width,
            height,
            20,
            3.5,
            generator=generator,
            clip_image=clip_image_pil,
            clip_image2=clip_image_pil2,
            face_emb=face_embs,
        ).images
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        res_image_pil_1 = Image.fromarray((image * 255).astype(np.uint8))
        
        # Save ref_image, src_image and the generated_image
        ref_image_pil = ref_image_pil.resize((w, h))
        pose_image_pil = pose_image_pil.resize((w, h))
        canvas.paste(ref_image_pil, (0 , jdx * h))
        canvas.paste(clip_image_pil.resize((512, 512)), (w , jdx * h))
        canvas.paste(pose_image_pil, (w *2, jdx*h))
        canvas.paste(res_image_pil_1, (w * 3, jdx*h))
        canvas.paste(gt_image_pil, (w * 4, jdx*h))
    save_dir = "./metadata/" + net_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    canvas.save(save_dir + "image_{}.png".format(num_iters))

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipe
    torch.cuda.empty_cache()

    return canvas 

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        cpu=(cfg.device == 'cpu'),
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
    ).to(device=cfg.device)
    denoising_unet = UNet3DMultiConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device=cfg.device)
    
    denoising_unet = set_ip_adapter(denoising_unet)

    if cfg.pose_guider_pretrain:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256), conditioning_channels=config.cond_chan,
        ).to(device=cfg.device)
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
        ).to(device=cfg.device)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    pose_guider.requires_grad_(True)

    num_tokens_image_proj = 16
    image_emb_dim_image_proj = 512
    
    image_proj_model =  Resampler(
        dim=1280,
        depth=4, 
        dim_head=64,
        heads=20,
        num_queries=num_tokens_image_proj,
        embedding_dim=image_emb_dim_image_proj,
        output_dim=denoising_unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(device=cfg.device)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = InstanceNet(
        reference_unet,
        denoising_unet,
        pose_guider,
        image_proj_model,
        reference_control_writer,
        reference_control_reader,
        device=cfg.device
    )
    
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    # Freeze
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        cfg.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,).to(dtype=weight_dtype, device=cfg.device)
    image_enc.requires_grad_(False)
    image_proj_model.requires_grad_(True)
    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # define train dataset and val dataset
    train_dataset, val_dataset = get_deepfashion_dataset(cfg, double_clip=True, use_face_emb=True)

    train_dataloader = torch.utils.data.DataLoader(
       train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
       val_dataset, batch_size=cfg.data.val_bs, shuffle=False, num_workers=2, pin_memory=True, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # here we write the training code
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # pdb.set_trace()
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["img"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                face_emb = batch["face_emb"].to(weight_dtype)
                
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                tgt_pose_img = batch["tgt_pose"]
                
                # face mask can directly get from the densepose estimation. Only used in training.
                face_mask = (tgt_pose_img[:, 2, :, :] * 255 == 23 ).unsqueeze(1) + (tgt_pose_img[:, 2, :, :] * 255 == 24 ).unsqueeze(1)
                h_latents = 64
                face_mask = F.interpolate(input=face_mask.to(dtype=torch.float), size=(h_latents, h_latents)) > 0
                tgt_pose_img = tgt_pose_img.unsqueeze(2)  # (bs, 3, 1, 512, 512)
                
                uncond_fwd = random.random() < cfg.uncond_ratio
                
                clip_image_list = []
                ref_image_list = []
                clip_image_list2 = []
                
                for batch_idx, (ref_img, clip_img, clip_img2) in enumerate(
                    zip(
                        batch["ref_img"],
                        batch["clip_images"],
                        batch["clip_images2"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                        clip_image_list2.append(torch.zeros_like(clip_img2))
                    else:
                        clip_image_list.append(clip_img)
                        clip_image_list2.append(clip_img2)
                    ref_image_list.append(ref_img)
                
                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    
                    # generate random noise of clip, to improve the robustness.
                    if random.random() < 0.1:
                        clip_img = torch.zeros(clip_img.shape, dtype=clip_img.dtype, device=clip_img.device)
                    clip_image_embeds = image_enc(
                        clip_img.to(cfg.device, dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                    
                    random_clip_embs = generate_clip_random_noise(bs=clip_image_embeds.shape[0], 
                                                                  device=clip_image_embeds.device, 
                                                                  dtype=clip_image_embeds.dtype)
                    image_prompt_embeds = image_prompt_embeds + random_clip_embs
                    
                    clip_img2 = torch.stack(clip_image_list2, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds2 = image_enc(
                        clip_img2.to(cfg.device, dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds2 = clip_image_embeds2.unsqueeze(1)  # (bs, 1, d)
                    
                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    face_emb,
                    image_prompt_embeds,
                    clip_image_embeds2=image_prompt_embeds2,
                    pose_img=tgt_pose_img,
                    uncond_fwd=uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    
                    loss1 = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    face_mask = face_mask.repeat(1, model_pred.shape[1], 1, 1)
                    face_mask = face_mask.unsqueeze(2)
                    loss2 = F.mse_loss(
                        model_pred.float() * face_mask.float(), target.float() * face_mask.float(), reduction="none",
                    )
                    loss = loss1 + loss2
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)

                # log validation
                if (global_step - 1) % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            data_config=cfg.data,
                            net_name=cfg.exp_name,
                            num_iters=global_step,
                            val_dataloader=val_dataloader,
                        )


            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            if not cfg.no_save:
                unwrap_net = accelerator.unwrap_model(net)
                save_checkpoint(
                    unwrap_net.reference_unet,
                    save_dir,
                    "reference_unet",
                    global_step,
                    total_limit=3,
                )
                save_checkpoint(
                    unwrap_net.denoising_unet,
                    save_dir,
                    "denoising_unet",
                    global_step,
                    total_limit=3,
                )
                save_checkpoint(
                    unwrap_net.pose_guider,
                    save_dir,
                    "pose_guider",
                    global_step,
                    total_limit=3,
                )
                save_checkpoint(
                    unwrap_net.image_proj_model,
                    save_dir,
                    "image_proj_model",
                    global_step,
                    total_limit=3,
                )


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/train.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '2'
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
