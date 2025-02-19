import inspect
from PIL import Image
import torch
import os
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .embeddings import align_control_guidance
from diffusers.utils import (
    PIL_INTERPOLATION,
)


def check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}."
            )


def check_image(image, prompt, prompt_embeds):
    image_is_pil = isinstance(image, Image.Image)
    image_is_tensor = isinstance(image, torch.Tensor)
    image_is_pil_list = isinstance(image, list) and isinstance(image[0], Image.Image)
    image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

    if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
        raise TypeError(
            "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
        )

    if image_is_pil:
        image_batch_size = 1
    elif image_is_tensor:
        image_batch_size = image.shape[0]
    elif image_is_pil_list:
        image_batch_size = len(image)
    elif image_is_tensor_list:
        image_batch_size = len(image)

    if prompt is not None and isinstance(prompt, str):
        prompt_batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        prompt_batch_size = len(prompt)
    elif prompt_embeds is not None:
        prompt_batch_size = prompt_embeds.shape[0]

    if image_batch_size != 1 and image_batch_size != prompt_batch_size:
        raise ValueError(
            f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
        )


def check_and_initialize(pipeline, prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds,
                        num_videos_per_prompt, video_length, frames, height, width, control_guidance_start, control_guidance_end, 
                        generator, eta, guidance_scale, **kwargs):
    # self.initialize_tracker()
    height, width = pipeline._default_height_width(height, width, frames)
    os.makedirs('vis/video_outputs', exist_ok=True)
    use_depth = pipeline.controlnet is not None

    pipeline.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    frames, depth_input = pipeline._prepare_images_and_depths(frames, kwargs.get("depths", None), batch_size=batch_size, 
                                        num_videos_per_prompt=num_videos_per_prompt, height=height, width=width, 
                                        device=device, do_classifier_free_guidance=do_classifier_free_guidance, 
                                        use_depth=use_depth, crop_coord=kwargs.get("crop_coord", None), 
                                        save_path=kwargs.get("save_path", None))

    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    controlnet_keep = align_control_guidance(pipeline, control_guidance_start, control_guidance_end) if use_depth else None

    latents = pipeline.prepare_video_latents(frames, batch_size, pipeline.weight_dtype, device, generator=generator)

    outputs = {'frames': frames, 'depth_input': depth_input, 'controlnet_keep': controlnet_keep, 'extra_step_kwargs': extra_step_kwargs,
               'latents': latents, 'device': device, 'do_classifier_free_guidance': do_classifier_free_guidance}
    return outputs


def decode_latents(pipeline, latents, return_tensor=False):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = pipeline.vae.decode(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    if return_tensor:
        return video
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


def prepare_extra_step_kwargs(pipeline, generator, eta):
    extra_step_kwargs = {}
    if "generator" in inspect.signature(pipeline.scheduler.step).parameters:
        extra_step_kwargs["generator"] = generator
    if "eta" in inspect.signature(pipeline.scheduler.step).parameters:
        extra_step_kwargs["eta"] = eta
    return extra_step_kwargs


def prepare_image(
        image, width, height, batch_size, num_videos_per_prompt, device, dtype, do_classifier_free_guidance, do_normalize=True,
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image[0], Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            if do_normalize:
                image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_videos_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    return image


def prepare_video_latents(pipeline, frames, batch_size, dtype, device, generator=None):
    if not isinstance(frames, (torch.Tensor, Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    frames = frames[0].to(device=device, dtype=dtype)
    frames = rearrange(frames, "c f h w -> f c h w" )

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if isinstance(generator, list):
        latents = [
            pipeline.vae.encode(frames[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0)
    else:
        latents = pipeline.vae.encode(frames).latent_dist.sample(generator)

    latents = pipeline.vae.config.scaling_factor * latents

    latents = rearrange(latents, "f c h w ->c f h w")

    return latents[None]


def get_slide_window_indices(video_length, window_size):
    assert window_size >=3
    key_frame_indices = np.arange(0, video_length, window_size-1).tolist()

    # Append last index
    if key_frame_indices[-1] != (video_length-1):
        key_frame_indices.append(video_length-1)

    slices = np.split(np.arange(video_length), key_frame_indices)
    inter_frame_list = []
    for s in slices:
        if len(s) < 2:
            continue
        inter_frame_list.append(s[1:].tolist())
    return key_frame_indices, inter_frame_list


def default_height_width(height, width, image):
    # NOTE: It is possible that a list of images have different
    # dimensions for each image, so just checking the first image
    # is not _exactly_ correct, but it is simple.
    while isinstance(image, list):
        image = image[0]

    if height is None:
        if isinstance(image, Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[3]

        height = (height // 8) * 8  # round down to nearest multiple of 8

    if width is None:
        if isinstance(image, Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[2]

        width = (width // 8) * 8  # round down to nearest multiple of 8

    return height, width


def batch_to_head_dim(tensor: torch.Tensor, head_size: int) -> torch.Tensor:
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor


def head_to_batch_dim(tensor: torch.Tensor, head_size: int, out_dim: int = 3) -> torch.Tensor:
    if tensor.ndim == 3:
        batch_size, seq_len, dim = tensor.shape
        extra_dim = 1
    else:
        batch_size, extra_dim, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)
    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)
    return tensor


def _prepare_image(pipeline, image, width, height, batch_size, num_videos_per_prompt, device, dtype, do_classifier_free_guidance, crop_coord=None, do_normalize=True):
    prepared_image = pipeline.prepare_image(
        image=image,
        width=width,
        height=height,
        batch_size=batch_size * num_videos_per_prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        device=device,
        dtype=dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
        do_normalize=do_normalize,
    )
    if crop_coord is not None:
        y_start, y_end, x_start, x_end = crop_coord
        cropped_image = prepared_image[:, :, y_start:y_end, x_start:x_end]
        prepared_image = F.interpolate(cropped_image, (512, 512), mode="bilinear", align_corners=False)
    return prepared_image


def prepare_images_and_depths(pipeline, frames, depth_total, batch_size, num_videos_per_prompt, height, width, device, 
                              do_classifier_free_guidance, use_depth, crop_coord=None, save_path=None, **kwargs):
        images = []
        depths = []

        for i, i_img in enumerate(frames):
            i_img = _prepare_image(pipeline, i_img, width, height, batch_size, num_videos_per_prompt, device, 
                pipeline.weight_dtype, do_classifier_free_guidance, crop_coord=crop_coord, do_normalize=True)
            images.append(i_img)

            if use_depth:
                depth = 1 - np.array(depth_total[i])  # because Marigold depth is inverted
                depth = (depth*255).astype(np.uint8)
                if depth.shape == 2:
                    depth = depth[:, :, None]
                    depth = np.concatenate([depth, depth, depth], axis=2)
                depth = Image.fromarray(depth)
                depth = _prepare_image(pipeline, depth, width, height, batch_size, num_videos_per_prompt, device,
                    pipeline.controlnet.dtype, do_classifier_free_guidance, crop_coord=crop_coord, do_normalize=False)
                _plot_image_and_depth(i_img, depth, show=False, save_path=save_path)
                depths.append(depth) 

        images = torch.stack(images, dim=2)
        depths = torch.stack(depths, dim=2) if use_depth else None
        return images, depths


def _plot_image_and_depth(image, depth, show=False, save_path=None):
    im1 = image[0].cpu().numpy().transpose(1, 2, 0)
    im2 = depth[0].cpu().numpy().transpose(1, 2, 0)
    im1 = (im1 + 1) / 2
    # im2 = (im2 + 1) / 2
    im1 = (im1 * 255).astype(np.uint8)
    im2 = (im2 * 255).astype(np.uint8)
    if show:
        plt.imshow(im1)
        plt.show()
        plt.imshow(im2)
        plt.show()
    if save_path is not None:
        im1 = Image.fromarray(im1)
        im2 = Image.fromarray(im2)
        im_path = os.path.join(save_path, 'input.png')
        depth_path = os.path.join(save_path, 'depth.png')
        im1.save(im_path)
        im2.save(depth_path)

