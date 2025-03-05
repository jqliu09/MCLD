import numpy as np
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline, Pose2ImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from typing import Callable, List, Optional, Union
import torch
from src.models.mutual_self_attention import ReferenceAttentionControl
from PIL import Image
import pdb
import torchvision.transforms as transforms
from ip_adapter.attention_processors import IPAttnProcessor2_0
from src.utils.attn_utils import process_attnmap
import cv2

class Pose2ImageIpMultiPipeline(Pose2ImagePipeline):
        
    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.image_proj_model = image_proj_model
        self.dino_process = transforms.Compose([
            transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        pose_image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        clip_image=None,
        face_emb=None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # ad
        
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # batch_size = 1
        try:
            batch_size = ref_image.shape[0]
        except:
            batch_size = 1

        # Prepare clip image embeds
        clip_input = clip_image if clip_image is not None else ref_image
        clip_image = self.clip_image_processor.preprocess(
            clip_input, return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        
        clip_image2 = kwargs["clip_image2"]
        clip_image2 = self.clip_image_processor.preprocess(
                clip_image2, return_tensors="pt" 
            ).pixel_values
        clip_image_embeds2 = self.image_encoder(
                clip_image2.to(device, dtype=self.image_encoder.dtype)
            ).image_embeds
        image_prompt_embeds2 = clip_image_embeds2.unsqueeze(1)
        uncond_image_prompt_embeds2 = torch.zeros_like(image_prompt_embeds2)

        # prepare face embs
        # pdb.set_trace()
        face_prompt_embeds = self.image_proj_model(face_emb)
        uncond_face_prompt_embeds = torch.zeros_like(face_prompt_embeds)
        
        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )
            face_prompt_embeds = torch.cat(
                [uncond_face_prompt_embeds, face_prompt_embeds], dim=0
            )
            image_prompt_embeds2 = torch.cat(
                [uncond_image_prompt_embeds2, image_prompt_embeds2], dim=0
            )

        # prepare the stage-wise conditions 
        down_encoder_prompt_embeds = torch.cat([image_prompt_embeds, face_prompt_embeds], dim=1)
        mid_encoder_prompt_embeds = torch.cat([image_prompt_embeds, image_prompt_embeds2, face_prompt_embeds], dim=1)
        up_encoder_prompt_embeds = torch.cat([image_prompt_embeds2, face_prompt_embeds], dim=1)
        
        
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            clip_image_embeds.dtype,
            device,
            generator,
        )
        latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215 # (b, 4, h, w)

        # Prepare pose condition image
        pose_cond_tensor = self.cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea = (
            torch.cat([pose_fea] * 2) if do_classifier_free_guidance else pose_fea
        )

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        if True:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # pdb.set_trace()
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    down_encoder_hidden_states=down_encoder_prompt_embeds,
                    mid_encoder_hidden_states=mid_encoder_prompt_embeds,
                    up_encoder_hidden_states=up_encoder_prompt_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )[0]

                if "return_attn" in kwargs.keys():
                # if True:
                    ip_attn_oi = []
                    ip_attn_tm = []
                    ip_attn_face = []
                    for attn_name, attn_module in self.denoising_unet.attn_processors.items():
                        if isinstance(attn_module, IPAttnProcessor2_0):
                            if 'up' in attn_name:
                                ip_attn_tm.append(torch.clone(attn_module.attn_map2))
                                ip_attn_face.append(torch.clone(attn_module.attn_map))
                                # print(attn_name)
                            elif 'down' in attn_name:
                                ip_attn_oi.append(torch.clone(attn_module.attn_map2))
                                # print(attn_name)
                    attn_oi = process_attnmap(ip_attn_oi, dst_size=[256, 256])
                    attn_tm = process_attnmap(ip_attn_tm, dst_size=[256, 256])
                    attn_face = process_attnmap(ip_attn_face, dst_size=[256, 256])
                    
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    # progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        image = self.decode_latents(latents)  # (b, c, 1, h, w)

        if "return_attn" in kwargs.keys():
        # if True:
            image_ = (image[0,:,0].transpose([1,2,0])*255).astype(np.uint8)
            image_ = cv2.resize(image_, (256, 256))
            attn_io_image = cv2.addWeighted(attn_oi, 0.6, image_, 0.4, 0)
            attn_tm_image = cv2.addWeighted(attn_tm, 0.6, image_, 0.4, 0)
            attn_face_image = cv2.addWeighted(attn_face, 0.6, image_, 0.4, 0)
            return_out = np.concatenate([attn_io_image, attn_tm_image, attn_face_image])
            return return_out
            
            
        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)

        if not return_dict:
            return image

        return Pose2ImagePipelineOutput(images=image)