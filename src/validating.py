import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torchvision
import torch.nn.functional as F
from .metrics import calculate_iou
from .frequency_filter import frequency_loss
from .was_attention import calculate_was_mask, calculate_multi_was_mask
from utils.image_processing import visualize_attention_maps, process_and_save_images
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .embeddings import align_control_guidance, calculate_controlnet_features


def validate_pipeline(
    pipeline,
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
    training_mode = kwargs.get("training_mode", "text_embeds")
    if pipeline.accelerator is not None:
        weight_dtype = torch.float32
        if pipeline.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif pipeline.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float16

    valid_output = validation(pipeline, prompt=prompt, video_length=video_length,
        height=height, width=width, guidance_scale=guidance_scale, epoch=0,
        attention_output_size=attention_output_size, token_ids=token_ids, average_layers=average_layers, 
        apply_softmax=apply_softmax, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, 
        eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, output_type=output_type, return_dict=return_dict, 
        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, 
        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start, 
        control_guidance_end=control_guidance_end, generate_new_noise=generate_new_noise, weight_dtype=weight_dtype,
        complete_dataset=True, #True if epoch % 20 == 19 else False,
        **kwargs,
    )

    ious = []
    iou_str = ""
    for part_name in pipeline.part_names:
        iou = valid_output[f"iou_{part_name}"]
        ious.append(iou)
        iou_str += f"{part_name}: {iou:.4f}, "
    iou_str += f"Mean: {np.mean(ious):.4f}"
    print(iou_str)


def validation(
    pipeline,
    prompt: Union[str, List[str]] = None,
    epoch: Optional[int] = 0,
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
    output_type: Optional[str] = "tensor",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    complete_dataset: Optional[bool] = False,
    generate_new_noise: Optional[bool] = True,
    weight_dtype: Optional[torch.dtype] = torch.float32,
    **kwargs,
):
    mode='valid'
    os.makedirs(f'vis/{mode}', exist_ok=True)
    t2i_transform = torchvision.transforms.ToPILImage()
    dataloader = pipeline.valid_dataloader
    
    pipeline.unet.eval()

    
    output = {f'iou_{part_name}':[] for part_name in pipeline.part_names}

    progress_bar = tqdm(total=len(dataloader), desc=f"Validation")
    for step, batch in enumerate(dataloader):
        if not complete_dataset and not (step <= 10):
            continue
        depth = batch.get("depth", None)
        use_depth = (depth is not None) and (pipeline.controlnet is not None)
        image, mask, prompt = batch["image"], batch["mask"], batch["prompt"]
        prompt = prompt[0]
        image = image[0, 0]#[[2, 1, 0]]
        image = t2i_transform((image*255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        image.save(f"vis/{mode}/input.png")
        frames = [image]
        depths = [depth[0].cpu().numpy()] if use_depth else None
        masks = [mask[0, 0]]
        prompt_embeds = None

        frames_full = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
            video_length=video_length, frames=frames, height=height, width=width,
            control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
            generator=generator, eta=eta, guidance_scale=guidance_scale, depths=depths)['frames']

        crop_coords = kwargs.get("crop_coords", None)
        if crop_coords is not None:
            maps = {}
            for crop_coord in crop_coords:
                map_ = validation_step(pipeline, prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt,
                            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                            num_videos_per_prompt=num_videos_per_prompt, video_length=video_length, frames=frames,
                            height=height, width=width, controlnet_conditioning_scale=controlnet_conditioning_scale,
                            control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
                            generator=generator, eta=eta, guidance_scale=guidance_scale, depths=depths, epoch=epoch, 
                            mode=mode, generate_new_noise=generate_new_noise, use_depth=use_depth, apply_softmax=apply_softmax,
                            attention_output_size=attention_output_size, token_ids=token_ids, average_layers=average_layers,
                            crop_coord=crop_coord, **kwargs)
                maps[f'{crop_coord}'] = map_
            was_attention = calculate_multi_was_mask(pipeline, maps, crop_coords, output_size=512, use_cross_attention_only=False)
        else:
            was_attention = validation_step(pipeline, prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt,
                        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                        num_videos_per_prompt=num_videos_per_prompt, video_length=video_length, frames=frames,
                        height=height, width=width, controlnet_conditioning_scale=controlnet_conditioning_scale,
                        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
                        generator=generator, eta=eta, guidance_scale=guidance_scale, depths=depths, epoch=epoch, 
                        mode=mode, generate_new_noise=generate_new_noise, use_depth=use_depth, apply_softmax=apply_softmax,
                        attention_output_size=attention_output_size, token_ids=token_ids, average_layers=average_layers,
                        **kwargs)['was_attention_maps']
        
        image, mask = frames_full[:, :, 0], masks[0].unsqueeze(0)
        # mask = F.interpolate(mask.unsqueeze(0), size=attention_output_size, mode="bilinear")[0]
        mask = mask[0]  

        one_shot_mask = (
            torch.zeros(pipeline.num_parts, mask.shape[0], mask.shape[1],).to(mask.device)
            .scatter_(0, mask.unsqueeze(0).type(torch.int64), 1.0))

        visualize_attention_maps(one_shot_mask.unsqueeze(0), output_path=f"./vis/{mode}/gt_mask",
            interpolation_size=512, attention_type="gt_mask", epoch=epoch, colorize=False)
        
        pred_mask = was_attention[0].argmax(0)
        for ind, part_name in enumerate(pipeline.part_names):
            gt = torch.where(mask == ind, 1, 0).type(torch.int64)
            pred = torch.where(pred_mask == ind, 1, 0).type(torch.int64)
            if torch.all(gt == 0):
                continue
            iou = calculate_iou(pred, gt)
            output[f"iou_{part_name}"] = np.append(output[f"iou_{part_name}"], iou.item())
        
        progress_bar.update(1)
        progress_bar.set_postfix({f"iou_{part_name}": output[f"iou_{part_name}"][-1] 
                                  if len(output[f"iou_{part_name}"]) > 0 else 0 for part_name in pipeline.part_names})

    for key in output:
        output[key] = output[key].mean() if len(output[key]) > 0 else 0

    return output


def validation_step(pipeline, prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, num_videos_per_prompt,
                    video_length, frames, height, width, controlnet_conditioning_scale, control_guidance_start, control_guidance_end, generator, 
                    eta, guidance_scale, depths, epoch, mode, generate_new_noise, use_depth, attention_output_size, token_ids, average_layers,
                    apply_softmax,
                    **kwargs):
    params = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=height, width=width,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, depths=depths, save_path=f"vis/valid", **kwargs)

    frames, depths, latents, controlnet_keep, extra_step_kwargs, device, do_classifier_free_guidance = (
        params['frames'], params['depth_input'], params['latents'], params['controlnet_keep'], 
        params['extra_step_kwargs'], params['device'], params['do_classifier_free_guidance']
        )

    training_mode = kwargs.get("training_mode", "text_embeds")
    prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        use_tokens="text_embeds" in training_mode,
    )

    pipeline.video_length = frames.shape[2]

    t = kwargs.get("time_interval", [5, 200])
    if mode == "valid":
        t = [95, 105]
    if len(t) == 2:
        t = torch.randint(t[0], t[1] + 1, [1], dtype=torch.long)
    t = t.to(device=device)

    with torch.no_grad():
        if generate_new_noise:
            noise = torch.randn_like(latents).to(device=device)
            pipeline.noise = noise
        else:
            noise = pipeline.noise.to(device)
        latents_noisy = pipeline.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy

        if use_depth:
            down_block_res_samples, mid_block_res_sample =\
            calculate_controlnet_features(pipeline, depth=depths, latent_model_input=latent_model_input,
                                        prompt_embeds=prompt_embeds, timestep=t, controlnet_keep=controlnet_keep,
                                        controlnet_conditioning_scale=controlnet_conditioning_scale,)
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        noise_pred_ = pipeline.unet(
            latent_model_input, t, encoder_hidden_states=prompt_embeds,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
        ).sample.to(device)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_

        _, sd_cross_attention_maps2, sd_self_attention_maps = pipeline.get_attention_map(
            pipeline.attention_maps, output_size=attention_output_size, token_ids=token_ids,
            average_layers=average_layers, apply_softmax=apply_softmax, guidance_scale=guidance_scale,
        )

    visualize_attention_maps(
        sd_cross_attention_maps2,
        output_path=f"./vis/{mode}/inference_cross_attention_maps2",
        interpolation_size=512,
        attention_type="cross_attention_maps",
        epoch=epoch,
        colorize=True,
    )

    was_attention_maps = \
        pipeline.calculate_was_mask(sd_cross_attention_maps2.clone(), sd_self_attention_maps.clone(),
                                            mask_patch_size=512, use_cross_attention_only=False)

    visualize_attention_maps(was_attention_maps, output_path=f"./vis/{mode}/inference_was_attention_maps2",
        attention_type="was_attention_maps",epoch=epoch, colorize=False)

    # process_and_save_images(was_attention_maps, frames, dataset_name=pipeline.dataset_name,
    #                                 output_dir=f'vis/{mode}/output', save_frames=False)

    pipeline.attention_maps = {}

    sd_cross_attention_maps2 = sd_cross_attention_maps2

    sd_cross_attention_maps2_softmax = sd_cross_attention_maps2[0].softmax(dim=0)
    small_sd_cross_attention_maps2 = F.interpolate(sd_cross_attention_maps2_softmax[None, ...], 
                                                                        64, mode="bilinear")[0]
    self_attention_map = (sd_self_attention_maps *
        small_sd_cross_attention_maps2.flatten(1, 2)[..., None, None]).sum(dim=1)

    was_attention = calculate_was_mask(pipeline, sd_cross_attention_maps2,
                    sd_self_attention_maps, mask_patch_size=512, use_cross_attention_only=False)[0]
    
    maps = {"cross_attention_maps": sd_cross_attention_maps2, "self_attention_maps": sd_self_attention_maps, 
        "was_attention_maps": was_attention_maps}
    
    return maps
