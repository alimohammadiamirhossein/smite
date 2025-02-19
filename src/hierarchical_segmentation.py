import os
import PIL
import cv2
import copy
import numpy as np
from tqdm import tqdm
from .inference import _visualize_attention_maps
import torch, torchvision
import torch.nn.functional as F
from scipy.signal import savgol_filter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def inference_with_hierarchical_segmentation(
        pipeline,
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
    frames_full = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=None, width=None, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, **kwargs)['frames']
    frames_full = frames_full.to('cpu')
    torch.cuda.empty_cache()
    depth_input = kwargs.get("depths", None)
    
    frames_512 = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=512, width=512, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, **kwargs)['frames']
    frames_512 = frames_512.to('cpu')
    torch.cuda.empty_cache()
    frames_512_list = []
    for i in range(frames_512.shape[2]):
        frames_512_list.append(frames_512[:, :, i].float())
    
    primitive_kwargs = copy.deepcopy(kwargs)
    primitive_kwargs['track_weight'], primitive_kwargs['regularization_weight'] = 0.0, 0.0
    primitive_kwargs['crop_coords'] = None
    was = pipeline(
        prompt, video_length, frames=frames_512_list, height=512, width=512, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, eta=eta,
        generator=generator, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        attention_output_size=attention_output_size, training_mode=training_mode, token_ids=token_ids,
        average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end, return_was=return_was, **primitive_kwargs)['segmentation']
    
    was = F.interpolate(was, (frames_full.shape[-2], frames_full.shape[-1]), mode='bilinear', align_corners=False)
    primitive_labels = was.argmax(dim=1)
    
    primitive_mask = torch.where(primitive_labels == 0, 0, 1)

    if frames_full.shape[-1] / frames_full.shape[-2] > 1.5 or frames_full.shape[-2] / frames_full.shape[-1] > 1.5:
       was_full, primitive_mask = handle_high_aspect_ratio_frames(frames_full, primitive_mask, was, pipeline,
                                        prompt, video_length, num_inference_steps, guidance_scale, negative_prompt,
                                        num_videos_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds,
                                        attention_output_size, training_mode, token_ids, average_layers, apply_softmax,
                                        output_type, return_dict, callback, callback_steps, cross_attention_kwargs,
                                        controlnet_conditioning_scale, control_guidance_start, control_guidance_end,
                                        return_was, primitive_kwargs, **kwargs)

    frames_full = frames_full.to('cuda')
    cropped_frames, information = crop_video_based_on_primitive_mask(frames_full, primitive_mask)
    # cropped_frames, information = crop_video_based_on_primitive_mask(frames_full, primitive_mask, expand_ratio=0.2) #for xmem
    del frames_512, frames_512_list, primitive_kwargs, primitive_labels, primitive_mask, was

    segmentations = pipeline(
        prompt, video_length, frames=cropped_frames, height=height, width=width, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, eta=eta,
        generator=generator, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        attention_output_size=attention_output_size, training_mode=training_mode, token_ids=token_ids,
        average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end, return_was=return_was, **kwargs)['segmentation']
    
    was_full = []
    for i in range(frames_full.shape[2]):
        if i not in information or i >= segmentations.shape[0]:
            print('you have skipping frame')
            continue
        x1, y1, x2, y2 = information[i]
        segmentation = segmentations[i]
        resized_segmentation = torch.nn.functional.interpolate(segmentation.unsqueeze(0), (y2-y1, x2-x1), mode='bilinear', align_corners=False).squeeze(0)
        was = torch.zeros((segmentations.shape[1], frames_full.shape[-2], frames_full.shape[-1]))
        was[0] = was[0] + segmentations.max()
        was[:, y1:y2, x1:x2] = resized_segmentation
        was_full.append(was)
    was_full = torch.stack(was_full, dim=0)
    maps_full = {'was_attention_maps': was_full, 'cross_attention_maps': None}
    _visualize_attention_maps(maps_full, frames_full, timestep=0, indice=1000, visualize_cross_attentions=False,
                              dataset_name=pipeline.dataset_name, output_path=kwargs.get('output_path', './output'))
    
    was_full = F.interpolate(was_full, (512, 512), mode='bilinear', align_corners=False)
    
    return was_full
    

def crop_video_based_on_primitive_mask(frames_full, primitive_mask, expand_ratio=0.04, depth_full=None):
    output = []
    information = {}

    if depth_full is not None:
        raise NotImplementedError("Depth is not implemented yet")

    for i in tqdm(range(primitive_mask.shape[0]), desc="Processing Hierarchical Segmentation"):
        mask = primitive_mask[i].cpu().numpy()
        indices = np.argwhere(mask == 1)
        
        if indices.size < 5:
            frame = frames_full[:, :, i]
            frame_new = F.interpolate(frame, (512, 512), mode='bilinear', align_corners=False)
            output.append(frame_new)
            information[i] = (0, 0, frame.shape[-1], frame.shape[-2])
        else:
            frame = frames_full[0, :, i]
            frame_height, frame_width = frame.shape[-2], frame.shape[-1]

            y1, x1 = np.min(indices, axis=0)  
            y2, x2 = np.max(indices, axis=0)  

            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            # Calculate dimensions and expand the bounding box
            width_ = x2 - x1
            height_ = y2 - y1
            x1_new = max(0, x1 - int(expand_ratio * width_))  # Expand to the left
            y1_new = max(0, y1 - int(expand_ratio * height_))  # Expand upwards
            x2_new = min(frame_width, x2 + int(expand_ratio * width_))  # Expand to the right
            y2_new = min(frame_height, y2 + int(expand_ratio * height_))  # Expand downwards

            # Crop the frame and resize
            frame = frame[:, y1_new:y2_new, x1_new:x2_new]
            frame = F.interpolate(frame.unsqueeze(0), (512, 512), mode='bilinear', align_corners=False)
            output.append(frame)
            information[i] = (x1_new, y1_new, x2_new, y2_new)
    
    return output, information


def handle_high_aspect_ratio_frames(
        frames_full,
        primitive_mask,
        was,
        pipeline,
        prompt,
        video_length,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        num_videos_per_prompt,
        eta,
        generator,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        attention_output_size,
        training_mode,
        token_ids,
        average_layers,
        apply_softmax,
        output_type,
        return_dict,
        callback,
        callback_steps,
        cross_attention_kwargs,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
        return_was,
        primitive_kwargs,
        **kwargs
):
    # Crop the image to a square region centered around the primitive_mask for each image and recalculate 'was'
    cropped_frames_list = []
    crop_positions = []
    skip_frames = []
    for i in range(frames_full.shape[2]):
        frame = frames_full[:, :, i]
        mask = primitive_mask[i]
        indices = torch.nonzero(mask)
        if indices.size(0) == 0:
            center_y = frames_full.shape[-2] // 2
            center_x = frames_full.shape[-1] // 2
            mask_height = frames_full.shape[-2]
            mask_width = frames_full.shape[-1]
        else:
            y_coords = indices[:, 0]
            x_coords = indices[:, 1]
            y_min = y_coords.min().item()
            y_max = y_coords.max().item()
            x_min = x_coords.min().item()
            x_max = x_coords.max().item()
            mask_height = y_max - y_min
            mask_width = x_max - x_min
            center_y = torch.mean(y_coords.float()).int()
            center_x = torch.mean(x_coords.float()).int()
        
        frame_height = frames_full.shape[-2]
        frame_width = frames_full.shape[-1]
        min_dim = min(frame_height, frame_width)
        if min_dim >= 512:
            crop_size = 512
        else:
            crop_size = min_dim
        half_size = crop_size // 2

        if mask_height > crop_size or mask_width > crop_size and primitive_mask[i].sum() / (mask_height * mask_width) > 0.5:
            skip_frames.append(i)

            cropped_frame = frame.float()
            x1, y1, x2, y2 = 0, 0, frame_width, frame_height

            resized_frame = F.interpolate(cropped_frame, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            x1 = center_x - half_size
            x2 = center_x + half_size
            y1 = center_y - half_size
            y2 = center_y + half_size

            if x1 < 0:
                x2 = x2 - x1
                x1 = 0
            if x2 > frame_width:
                x1 = x1 - (x2 - frame_width)
                x2 = frame_width
            if y1 < 0:
                y2 = y2 - y1
                y1 = 0
            if y2 > frame_height:
                y1 = y1 - (y2 - frame_height)
                y2 = frame_height

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            crop_width = x2 - x1
            crop_height = y2 - y1
            if crop_width != crop_size:
                x2 = x1 + crop_size
            if crop_height != crop_size:
                y2 = y1 + crop_size

            if x2 > frame_width:
                x1 = frame_width - crop_size
                x2 = frame_width
            if y2 > frame_height:
                y1 = frame_height - crop_size
                y2 = frame_height
            if x1 < 0:
                x1 = 0
                x2 = crop_size
            if y1 < 0:
                y1 = 0
                y2 = crop_size

            cropped_frame = frame[:, :, y1:y2, x1:x2].float()
            resized_frame = F.interpolate(cropped_frame, size=(512, 512), mode='bilinear', align_corners=False)
        
        cropped_frames_list.append(resized_frame)
        crop_positions.append((x1, y1, x2, y2))
    
    cropped_frames_tensor = torch.stack(cropped_frames_list, dim=2)

    primitive_kwargs = copy.deepcopy(primitive_kwargs)
    primitive_kwargs['track_weight'], primitive_kwargs['regularization_weight'] = 0.0, 0.0

    processed_frames_indices = [i for i in range(frames_full.shape[2]) if i not in skip_frames]
    if processed_frames_indices:
        frames_to_process = cropped_frames_tensor[:,:,processed_frames_indices]
        frames_to_process_list = [frames_to_process[:, :, i] for i in range(frames_to_process.shape[2])]
        was_processed = pipeline(
            prompt, video_length, frames=frames_to_process_list, height=512, width=512, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, eta=eta,
            generator=generator, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            attention_output_size=attention_output_size, training_mode=training_mode, token_ids=token_ids,
            average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
            callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end, return_was=return_was, **primitive_kwargs)['segmentation']
    else:
        was_processed = None

    was_full = []
    was_index = 0
    for i in range(frames_full.shape[2]):
        x1, y1, x2, y2 = crop_positions[i]
        if i in skip_frames:
            was_frame = was[i]
        else:
            segmentation = was_processed[was_index]
            was_index += 1
            segmentation = F.interpolate(segmentation.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
            was_frame = torch.zeros((segmentation.shape[0], frames_full.shape[-2], frames_full.shape[-1]), device=segmentation.device)
            was_frame[:, y1:y2, x1:x2] = segmentation
        was_full.append(was_frame)
    was_full = torch.stack(was_full, dim=0)
    primitive_labels = was_full.argmax(dim=1)
    primitive_mask = torch.where(primitive_labels == 0, 0, 1)

    return was_full, primitive_mask
