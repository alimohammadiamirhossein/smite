import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch
import math
import torchvision
from torch import optim
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL
from diffusers import ModelMixin
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    BaseOutput
)
# from models.util import process_and_save_images, visualize_attention_maps
from models import ControlNet3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from einops import rearrange
import torch.nn.functional as F
from models import UNet3DConditionModel
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from tqdm import tqdm

from .tracking import initialize_tracker, apply_tracker, apply_tracker_on_all_frames
from .hooks import remove_hook, create_hook, add_perturb_hook
from .optimization import (
                           configure_optimizers,
                           enable_accelerator,
                           )
from .data_processing import (
                              check_inputs,
                              check_image,
                              decode_latents,
                              prepare_image,
                              batch_to_head_dim,
                              head_to_batch_dim,
                              default_height_width,
                              check_and_initialize,
                              prepare_video_latents,
                              get_slide_window_indices,
                              prepare_extra_step_kwargs,
                              prepare_images_and_depths,
                              )
from .embeddings import set_embeddings_to_optimize, encode_prompt, calculate_controlnet_features
from .inference import _inference, inference, inference_with_modulation
from .hierarchical_segmentation import inference_with_hierarchical_segmentation
from .unet_utils import clean_features, inject_layers, load_weights
from .metrics import calculate_metrics
from .scheduler_utils import (
                                get_alpha_prev,
                                get_inverse_timesteps,
                               )
from .was_attention import get_attention_map, calculate_was_mask
from .latent_optimization import optimize_latents
from .training import train_pipeline
from .validating import validate_pipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class SMITEPipelineOutput(BaseOutput):
    segmentation: Union[torch.Tensor, np.ndarray]


class SMITEPipeline(DiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet3DConditionModel,
            controlnet: ControlNet3DModel,
            scheduler: DDIMScheduler,
            inverse_scheduler: DDIMInverseScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            inverse_scheduler=inverse_scheduler
        )
        self._initialize_attributes()

    def _initialize_attributes(self):
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.attention_maps = {}
        self.noise = None
        self.handles = None
        self.video_length = None
        self.num_parts = None
        self.embeddings_to_optimize = None
        self.accelerator = None
        self.learning_rate = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.optimizer = None
        self.part_names = None
        self.num_of_parts = None

    def _initialize_logger(self, dataset_name='pascal'):
        self.dataset_name = dataset_name
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.writer = SummaryWriter(f'logs/log_{dataset_name}_{current_time}')
        current_dir = os.getcwd()
        self.writer_path = f'{os.getcwd()}/logs/log_{dataset_name}_{current_time}'
    
    def initialize_tracker(self):
        initialize_tracker(self)

    def set_attention_hook(self, attention_layers_to_use: List[str]):
        create_hook(self, attention_layers_to_use)

    def add_perturb_hook(self, modification):
        add_perturb_hook(self, modification)

    def remove_hook(self):
        remove_hook(self)

    def enable_accelerator(self, mixed_precision: Optional[str] = 'fp16', train_dataloader: Optional[torch.utils.data.DataLoader] = None, 
                           valid_dataloader: Optional[torch.utils.data.DataLoader] = None):
        enable_accelerator(self, mixed_precision, train_dataloader, valid_dataloader)

    def configure_optimizers(self, learning_rate: float = 1e-4, training_mode="text_embeds"):
        configure_optimizers(self, learning_rate, training_mode)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def set_embeddings_to_optimize(self, num_of_part, embedding_initialize_prompt=None):
        set_embeddings_to_optimize(self, num_of_part, embedding_initialize_prompt)

    def load_weights(self, loading_path=None):
        load_weights(self, loading_path=loading_path)

    def _encode_prompt(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            use_tokens: bool = False,
    ):
        return encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt,
                             prompt_embeds, negative_prompt_embeds, use_tokens)

    def decode_latents(self, latents, return_tensor=False):
        return decode_latents(self, latents, return_tensor)

    def prepare_extra_step_kwargs(self, generator, eta):
        return prepare_extra_step_kwargs(self, generator, eta)

    def check_inputs(self, prompt, height, width, callback_steps, negative_prompt=None,
                     prompt_embeds=None, negative_prompt_embeds=None):
        check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    def check_image(self, image, prompt, prompt_embeds):
        check_image(image, prompt, prompt_embeds)
    
    def check_and_initialize(self, prompt, callback_steps, negative_prompt, prompt_embeds, \
                            negative_prompt_embeds, num_videos_per_prompt, video_length, frames, 
                            height, width, control_guidance_start, control_guidance_end, 
                            generator, eta, guidance_scale, **kwargs):
        return check_and_initialize(self, prompt, callback_steps, negative_prompt, prompt_embeds, \
                            negative_prompt_embeds, num_videos_per_prompt, video_length, frames, 
                            height, width, control_guidance_start, control_guidance_end, 
                            generator, eta, guidance_scale, **kwargs)

    def prepare_image(
            self, image, width, height, batch_size, num_videos_per_prompt, device, dtype, do_classifier_free_guidance, do_normalize=True,
    ):
        return prepare_image(image, width, height, batch_size, num_videos_per_prompt, device, dtype, do_classifier_free_guidance, do_normalize)
    
    def prepare_video_latents(self, frames, batch_size, dtype, device, generator=None):
        return prepare_video_latents(self, frames, batch_size, dtype, device, generator)

    def _default_height_width(self, height, width, image):
        return default_height_width(height, width, image)

    def get_alpha_prev(self, timestep):
        return get_alpha_prev(self, timestep)

    def get_slide_window_indices(self, video_length, window_size):
        return get_slide_window_indices(video_length, window_size)

    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        return get_inverse_timesteps(self, num_inference_steps, strength, device)

    def get_attention_map(self, raw_attention_maps, output_size=256, token_ids=(2,), average_layers=True,
                          train=True, apply_softmax=True, video_length=1, guidance_scale=0.0, **kwargs):
        return get_attention_map(self, raw_attention_maps, output_size, token_ids, average_layers, train,
                                 apply_softmax, video_length, guidance_scale, **kwargs)
    
    def apply_tracker(self, frames, query_frame=0, interpolate_size=64, batching_ratio=None):
        return apply_tracker(self, frames, query_frame=query_frame, interpolate_size=interpolate_size, 
                             batching_ratio=batching_ratio)
    
    def apply_tracker_on_all_frames(self, frames):
        return apply_tracker_on_all_frames(self, frames)
    
    def calculate_was_mask(self, sd_cross_attention_maps2_all_frames, sd_self_attention_maps,
                                          mask_patch_size=64, use_cross_attention_only=False):
        return calculate_was_mask(self, sd_cross_attention_maps2_all_frames, sd_self_attention_maps,
                                                mask_patch_size=mask_patch_size, use_cross_attention_only=use_cross_attention_only)
    
    def fit(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            height: Optional[int] = None,
            width: Optional[int] = None,
            guidance_scale: float = 7.5,
            attention_output_size: Optional[int] = 256,
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_of_epochs: Optional[int] = 300,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            generate_new_noise: Optional[bool] = True,
            **kwargs,
    ):
        return train_pipeline(pipeline=self, prompt=prompt, video_length=video_length, height=height, width=width,
                             guidance_scale=guidance_scale, attention_output_size=attention_output_size, token_ids=token_ids, 
                             average_layers=average_layers, apply_softmax=apply_softmax, negative_prompt=negative_prompt, 
                             num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, 
                             prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                             num_of_epochs=num_of_epochs, output_type=output_type, return_dict=return_dict, 
                             callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, 
                             controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                             control_guidance_end=control_guidance_end, generate_new_noise=generate_new_noise, **kwargs)
    
    def validate(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            height: Optional[int] = None,
            width: Optional[int] = None,
            guidance_scale: float = 7.5,
            attention_output_size: Optional[int] = 256,
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_of_epochs: Optional[int] = 300,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            generate_new_noise: Optional[bool] = True,
            **kwargs,
    ):
        return validate_pipeline(pipeline=self, prompt=prompt, video_length=video_length, height=height, width=width,
                             guidance_scale=guidance_scale, attention_output_size=attention_output_size, token_ids=token_ids, 
                             average_layers=average_layers, apply_softmax=apply_softmax, negative_prompt=negative_prompt, 
                             num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, 
                             prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                             num_of_epochs=num_of_epochs, output_type=output_type, return_dict=return_dict, 
                             callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, 
                             controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                             control_guidance_end=control_guidance_end, generate_new_noise=generate_new_noise, **kwargs)

    def calculate_controlnet_features(
            self,
            depth: torch.FloatTensor,
            latent_model_input: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            timestep: Optional[Union[torch.Tensor, float, int]] = None,
            controlnet_keep: Optional[torch.FloatTensor] = None,
            controlnet_conditioning_scale: Optional[torch.FloatTensor] = None,
    ):
        return calculate_controlnet_features(self, depth, latent_model_input, prompt_embeds, 
                                                  timestep, controlnet_keep, controlnet_conditioning_scale)

    def _prepare_images_and_depths(self, frames, depths, batch_size, num_videos_per_prompt, height, width, device, 
                              do_classifier_free_guidance, use_depth, crop_coord=None, save_path=None):
        return prepare_images_and_depths(self, frames, depths, batch_size, num_videos_per_prompt, height, width, device, 
                              do_classifier_free_guidance, use_depth, crop_coord=crop_coord, save_path=save_path)
    
    def clean_features(self):
        clean_features(self)
    
    def inject_layers(self, saved_features, i, guidance_scale, slice_=None):
        inject_layers(self, saved_features, i, guidance_scale=guidance_scale, slice_=slice_)
    
    def optimize_latents(self, 
                      prompt_embeds: torch.FloatTensor,
                      latents: torch.FloatTensor, 
                      saved_features: Dict[str, List[torch.FloatTensor]],
                      was_refrence: torch.FloatTensor, 
                      was_current: torch.FloatTensor,
                      t: int,
                      depth_input: Optional[torch.FloatTensor]=None,
                      guidance_scale: Optional[float]=7.5,
                      attention_output_size: Optional[int]=256,
                      token_ids: Optional[Union[List[int], Tuple[int]]]=None,
                      average_layers: Optional[bool]=True,
                      apply_softmax: Optional[bool]=True,
                      video_length: Optional[int]=1,
                      cross_attention_kwargs: Optional[Dict[str, Any]]=None,
                      controlnet_keep: Optional[torch.FloatTensor]=None,
                      controlnet_conditioning_scale: Optional[Union[float, List[float]]]=1.0,
                      extra_step_kwargs: Optional[Dict[str, Any]]=None,
                      **kwargs):
        return optimize_latents(self, prompt_embeds=prompt_embeds, latents=latents, saved_features=saved_features,
                                was_refrence=was_refrence, was_current=was_current, t=t, depth_input=depth_input,
                                guidance_scale=guidance_scale, attention_output_size=attention_output_size,
                                token_ids=token_ids, average_layers=average_layers, apply_softmax=apply_softmax,
                                video_length=video_length, cross_attention_kwargs=cross_attention_kwargs,
                                controlnet_keep=controlnet_keep, controlnet_conditioning_scale=controlnet_conditioning_scale,
                                extra_step_kwargs=extra_step_kwargs, **kwargs)

    def calculate_metrics(self, y_true, y_pred, gt_inds=None, frame_inds=None, output_path=None):
        return calculate_metrics(self, y_true, y_pred, gt_inds=gt_inds, frame_inds=frame_inds, output_path=output_path)
    
    def _inference(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            frames: Union[List[torch.FloatTensor], List[PIL.Image.Image], List[List[torch.FloatTensor]], List[List[PIL.Image.Image]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            attention_output_size: Optional[int] = 256,
            training_mode: Optional[str] = "text_embeds",
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            return_was: bool = False,
            **kwargs,
    ):
        return _inference(self, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                   num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                   num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                   negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                   average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                   callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
                   controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                   control_guidance_end=control_guidance_end, training_mode=training_mode, return_was=return_was, **kwargs)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            frames: Union[List[torch.FloatTensor], List[PIL.Image.Image], List[List[torch.FloatTensor]], List[List[PIL.Image.Image]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            attention_output_size: Optional[int] = 256,
            training_mode: Optional[str] = "text_embeds",
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            **kwargs,
    ):
        video_segmentation = inference(self, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                        num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                        average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
                        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                        control_guidance_end=control_guidance_end, training_mode=training_mode, **kwargs)
        if video_segmentation is None:
            return None
        if not return_dict:
            return video_segmentation
        return SMITEPipelineOutput(segmentation=video_segmentation)

    @torch.no_grad()
    def inference_with_modulation(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            frames: Union[List[torch.FloatTensor], List[PIL.Image.Image], List[List[torch.FloatTensor]], List[List[PIL.Image.Image]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            attention_output_size: Optional[int] = 256,
            training_mode: Optional[str] = "text_embeds",
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            **kwargs,
            ):
        video_segmentation = inference_with_modulation(self, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                          num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                          num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                          negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                          average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                          callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, training_mode=training_mode,
                          controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                          control_guidance_end=control_guidance_end, **kwargs)
        if video_segmentation is None:
            return None
        if not return_dict:
            return video_segmentation
        return SMITEPipelineOutput(segmentation=video_segmentation)

    @torch.no_grad()
    def inference_with_hierarchical_segmentation(
            self,
            prompt: Union[str, List[str]] = None,
            video_length: Optional[int] = 1,
            frames: Union[List[torch.FloatTensor], List[PIL.Image.Image], List[List[torch.FloatTensor]], List[List[PIL.Image.Image]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            attention_output_size: Optional[int] = 256,
            training_mode: Optional[str] = "text_embeds",
            token_ids: Union[List[int], Tuple[int]] = None,
            average_layers: Optional[bool] = True,
            apply_softmax: Optional[bool] = True,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            **kwargs,
    ):
        video_segmentation = inference_with_hierarchical_segmentation(self, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                        num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                        average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
                        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                        control_guidance_end=control_guidance_end, training_mode=training_mode, **kwargs)
        if video_segmentation is None:
            return None
        if not return_dict:
            return video_segmentation
        return SMITEPipelineOutput(segmentation=video_segmentation)
