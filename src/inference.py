import torch, torchvision
import os
import PIL
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .frequency_filter import dct_loss
from .was_attention import calculate_multi_was_mask
from .embeddings import calculate_controlnet_features
from utils.image_processing import process_and_save_images, visualize_attention_maps
from .slicing import create_slices, create_slices2, _apply_slicing_with_crops, _apply_slicing_without_crops

#TODO remove this function
def save_attention_maps(maps, frames, output_path, timestep=-1):
    cross_attention_maps = maps["cross_attention_maps"]
    self_attention_maps = maps["self_attention_maps"]
    print(cross_attention_maps.shape, self_attention_maps.shape)
    output_dir=f'{output_path}/Final_Output/output_{timestep}'
    torch.save(cross_attention_maps, f"{output_dir}_cross_attention_maps.pt")
    torch.save(self_attention_maps, f"{output_dir}_self_attention_maps.pt")
    torch.save(frames, f"{output_dir}_frames.pt")

def inference(
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
        video_length=video_length, frames=frames, height=height, width=width, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, **kwargs)['frames']
    frames_full = frames_full.to('cpu')
    torch.cuda.empty_cache()
    depth_input = kwargs.get("depths", None)
    
    # visualize_ground_truth(pipeline, frames_full, gt_=kwargs['gt'], output_path=kwargs.get('output_path', './output'))
    args_to_pass = {
                    "prompt": prompt, "height": height, "width": width, "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale, "negative_prompt": negative_prompt, "num_videos_per_prompt": num_videos_per_prompt,
                    "eta": eta, "generator": generator, "latents": latents, "prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds,
                    "attention_output_size": attention_output_size, "training_mode": training_mode, "token_ids": token_ids,
                    "average_layers": average_layers, "apply_softmax": apply_softmax, "output_type": output_type, "return_dict": return_dict,
                    "callback": callback, "callback_steps": callback_steps, "cross_attention_kwargs": cross_attention_kwargs,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale, "control_guidance_start": control_guidance_start,
                    "control_guidance_end": control_guidance_end, "return_was": True
                    }
    
    if kwargs.get("track_weight") or kwargs.get("regularization_weight"):
        if pipeline.controlnet is not None:
            slices = create_slices2(frames_full.shape[2], 16, overlap=8)
        else:
            slices = create_slices2(frames_full.shape[2], 16, overlap=8)
    else:
        slices = create_slices2(frames_full.shape[2], 18, overlap=0)

    crop_coords = kwargs.get("crop_coords", None)
    if crop_coords is not None:
        maps = {}
        for crop_coord in crop_coords:
            map_crop = _apply_slicing_with_crops(slices, frames, depth_input, pipeline, crop_coord, args_to_pass, **kwargs)
            maps[f'{crop_coord}'] = map_crop
        was_final = calculate_multi_was_mask(pipeline, maps, crop_coords, output_size=512, use_cross_attention_only=False)
        maps_full = {'was_attention_maps': was_final, 'cross_attention_maps': None}
    else:
        maps_full = _apply_slicing_without_crops(slices, frames, depth_input, pipeline, args_to_pass, **kwargs)
        was_final = maps_full['was_attention_maps']
        # save_attention_maps(maps_full, frames_full, output_path=kwargs.get('output_path', './output'), timestep=0)
    _visualize_attention_maps(maps_full, frames_full, timestep=0, indice=1000, visualize_cross_attentions=False,
                              dataset_name=pipeline.dataset_name, output_path=kwargs.get('output_path', './output'))
    return was_final

def _inference(
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
            ddim_inversion: bool = False,
            return_was: bool = False,
            **kwargs,
    ):
    saved_ref_features = {layer: [] for layer in kwargs["layers_to_save"]}
    ref_maps, saved_ref_features = _get_refrence(pipeline, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                          num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                          num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                          negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                          average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                          callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
                          training_mode=training_mode, saved_features=saved_ref_features,
                          controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                          control_guidance_end=control_guidance_end, return_was=True, **kwargs)

    params = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=height, width=width, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, **kwargs)

    frames, depth_input, latents, controlnet_keep, extra_step_kwargs, device, do_classifier_free_guidance = (
        params['frames'], params['depth_input'], params['latents'], params['controlnet_keep'], 
        params['extra_step_kwargs'], params['device'], params['do_classifier_free_guidance']
        )
    frames = frames.to(pipeline.device)
    if kwargs.get("track_weight") or kwargs.get("regularization_weight"):
        pred_tracks_indices, pred_visibilities = pipeline.apply_tracker_on_all_frames(frames)
    
    prompt_embeds_empty = pipeline._encode_prompt("", device, num_videos_per_prompt, 
        do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=None, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
    )

    saved_features = {layer: [] for layer in kwargs["layers_to_save"]}
    if ddim_inversion:
        STOPTIME = 0
        latents, saved_features, timesteps, kwargs = _ddim_inverse(pipeline, latents, prompt_embeds_empty, saved_features=saved_features,
                                                    num_inference_steps=num_inference_steps, device=device, STOPTIME=STOPTIME, 
                                                    do_classifier_free_guidance=do_classifier_free_guidance, 
                                                    cross_attention_kwargs=cross_attention_kwargs, 
                                                    extra_step_kwargs=extra_step_kwargs, **kwargs)
    else:
        timesteps, _, num_warmup_steps, kwargs = set_inference_timesteps(pipeline, "fast", device, **kwargs)
        timesteps += 100
        noise = torch.randn_like(latents).to(device=device)
        latents = pipeline.scheduler.add_noise(latents, noise, timesteps[0])

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order

    prompt_embeds_tokens = pipeline._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            use_tokens="text_embeds" in training_mode,
        )
    
    # if kwargs.get("track_weight", 0.0)==0 and kwargs.get("regularization_weight", 0.0)==0:
    #     return  {'maps': ref_maps, 'frames': frames}
    
    inference_args = {
        "prompt_embeds": prompt_embeds_tokens, "depth_input": depth_input,
        "attention_output_size": attention_output_size, "token_ids": token_ids, "average_layers": average_layers,
        "apply_softmax": apply_softmax, "video_length": video_length, "guidance_scale": guidance_scale,
        "cross_attention_kwargs": cross_attention_kwargs, "controlnet_keep": controlnet_keep,
        "controlnet_conditioning_scale": controlnet_conditioning_scale, "extra_step_kwargs": extra_step_kwargs,
    }
    
    if kwargs.get("track_weight", 0.0)==0 and kwargs.get("regularization_weight", 0.0)==0:
                return  {'maps': ref_maps, 'frames': frames}
    
    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            # print('loaded weights for time step', t, i, len(timesteps))
            torch.cuda.empty_cache()
            if ddim_inversion and t > STOPTIME:
                # print('loaded weights for time step', t, i)
                # pipeline.inject_layers(saved_features, i, guidance_scale)
                pipeline.inject_layers(saved_ref_features, 0, guidance_scale)
            else:
                pipeline.clean_features()
            with torch.no_grad():
                latents, maps = _process_timestep(pipeline, t=t, latents=latents, **inference_args, **kwargs)
                                
            # visualizing per step
            # _visualize_attention_maps(maps, frames, timestep=t, indice=i, visualize_cross_attentions=False,
            #                           dataset_name=pipeline.dataset_name, output_path=kwargs.get('output_path', './output'))

            if i < len(timesteps)-1:
                latents = pipeline.optimize_latents(saved_features=saved_features, latents=latents,
                                 t=t, was_refrence=ref_maps["was_64_attention_maps"], 
                                 i=i, was_current=maps["was_64_attention_maps"], 
                                 pred_tracks_indices=pred_tracks_indices, pred_visibilities=pred_visibilities,
                                 regularization_weight=kwargs.get("regularization_weight", 0.0),
                                 track_weight=kwargs.get("track_weight", 0.0),
                                 dct_threshold=kwargs.get("dct_threshold", 0.3),
                                 output_path=kwargs.get('output_path', './output'), **inference_args
                                 )
            # _was_mini(pipeline, maps, frames, save=True)
            progress_bar.update()
        if return_was:
            output_ = {'maps': maps, 'frames': frames}
            return output_
        video = pipeline.decode_latents(latents.to(pipeline.weight_dtype))
        return video


def inference_with_modulation(
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
        **kwargs,
):
    outputs_ = _inference(pipeline, prompt=prompt, video_length=video_length, frames=frames, height=height, width=width,
                     num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
                     num_videos_per_prompt=num_videos_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
                     negative_prompt_embeds=negative_prompt_embeds, attention_output_size=attention_output_size, token_ids=token_ids,
                     average_layers=average_layers, apply_softmax=apply_softmax, output_type=output_type, return_dict=return_dict,
                     callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs,
                     controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start,
                     control_guidance_end=control_guidance_end, return_was=True, **kwargs)

    unmodified_maps = outputs_['maps']
    # self.initialize_tracker()
    params = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=height, width=width, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=guidance_scale, **kwargs)

    frames, depth_input, latents, controlnet_keep, extra_step_kwargs, device, do_classifier_free_guidance = (
        params['frames'], params['depth_input'], params['latents'], params['controlnet_keep'], 
        params['extra_step_kwargs'], params['device'], params['do_classifier_free_guidance']
        )

    prompt_embeds_empty = pipeline._encode_prompt("", device, num_videos_per_prompt,
                                                  do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=None,
                                                  prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                                  )

    saved_features = {layer: [] for layer in kwargs["layers_to_save"]}
    STOPTIME = 0

    latents, saved_features, timesteps, kwargs = _ddim_inverse(pipeline, latents, prompt_embeds_empty, saved_features=saved_features,
                                                               num_inference_steps=num_inference_steps, device=device, STOPTIME=STOPTIME,
                                                               do_classifier_free_guidance=do_classifier_free_guidance,
                                                               cross_attention_kwargs=cross_attention_kwargs, inference_mode='modulation',
                                                               extra_step_kwargs=extra_step_kwargs, **kwargs)

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order

    prompt_embeds_tokens = pipeline._encode_prompt(
        "",
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        use_tokens="text_embeds" in training_mode,
    )

    was_mini = _was_mini(pipeline, unmodified_maps, frames)

    output_path = kwargs.get('output_path', './output')
    os.makedirs(f'{output_path}/modulation', exist_ok=True)

    diffs = modulation_loop(pipeline, latents, frames, saved_features, timesteps, video_length, guidance_scale, was_mini,
                            unmodified_maps['was_attention_maps'], prompt_embeds_tokens, depth_input, attention_output_size,
                            token_ids, average_layers, apply_softmax, cross_attention_kwargs, controlnet_keep,
                            controlnet_conditioning_scale, extra_step_kwargs, STOPTIME, **kwargs)

    process_and_save_images(diffs, frames, output_dir=f'{output_path}/modulation/output', save_frames=False)
    return diffs


def modulation_loop(pipeline, latents, frames, saved_features, timesteps, video_length, guidance_scale, was_mini, was_source,
                    prompt_embeds_tokens, depth_input, attention_output_size, token_ids, average_layers, apply_softmax,
                    cross_attention_kwargs, controlnet_keep, controlnet_conditioning_scale, extra_step_kwargs,
                    STOPTIME, **kwargs
                    ):
    org_latents = latents.clone()
    use_argmax = True
    if use_argmax:
        was_source = was_source[0].argmax(dim=0).to('cpu')
    else:
        was_source = was_source[0].softmax(dim=0).to('cpu')

    diffs = []
    with pipeline.progress_bar(total=len(timesteps)*2*pipeline.num_parts) as progress_bar:
        for p in range(pipeline.num_parts):
            videos = []
            for k in range(2):
                latents = org_latents.clone()
                got_perturbed, removed_hook = False, False
                for i, t in enumerate(timesteps):
                    torch.cuda.empty_cache()
                    if t > STOPTIME:
                        # print('loaded weights for time step', t)
                        pipeline.inject_layers(saved_features, i, guidance_scale)
                    else:
                        pipeline.clean_features()
                    if t < 300 and got_perturbed is False:
                        if removed_hook is False:
                            mul = 10. if k == 0 else -10.
                            mask = was_mini.argmax(dim=1)
                            mask = torch.where(mask == p, mul, 0)
                            mask = mask.reshape(mask.shape[0], -1)
                            pipeline.add_perturb_hook(mask)
                            removed_hook = True
                        else:
                            pipeline.remove_hook()
                            got_perturbed = True

                    latents, maps = _process_timestep(pipeline, prompt_embeds_tokens, latents, depth_input=depth_input,
                                                      t=t,  attention_output_size=attention_output_size,  token_ids=token_ids, average_layers=average_layers,
                                                      apply_softmax=apply_softmax, video_length=video_length, guidance_scale=guidance_scale,
                                                      cross_attention_kwargs=cross_attention_kwargs, controlnet_keep=controlnet_keep,
                                                      controlnet_conditioning_scale=controlnet_conditioning_scale, extra_step_kwargs=extra_step_kwargs,
                                                      **kwargs)
                    _visualize_attention_maps(maps, frames, timestep=t, indice=i, visualize_cross_attentions=False,
                                              dataset_name=pipeline.dataset_name, output_path=kwargs.get('output_path', './output'))
                    # _was_mini(pipeline, maps, frames, save=True)
                    progress_bar.update()
                video = pipeline.decode_latents(latents.to(pipeline.weight_dtype))
                videos.append(torch.from_numpy(video[0]))
            for idx in range(len(videos)):
                videos[idx] = torchvision.transforms.functional.gaussian_blur(videos[idx], kernel_size=7)
            seg = torch.linalg.norm(videos[1] - videos[0], dim=0)
            if use_argmax:
                source = torch.where(was_source == p, 1, 0)
            else:
                source = was_source[p]
            seg = seg * source + 0.6 * (1 - source) * seg
            # ################# thresholding #################
            # seg = torch.where(seg > seg.median(), seg, 0)
            # ################################################
            seg_image = 255*(seg[0]-seg[0].min())/(seg[0].max()-seg[0].min())
            output_path = kwargs.get('output_path', './output')
            PIL.Image.fromarray(np.uint8(seg_image.cpu().numpy())).save(f'{output_path}/modulation/seg_{p}.png')
            diffs.append(seg)
    diffs = torch.stack(diffs)
    diffs = diffs.permute(1, 0, 2, 3)
    return diffs


def _was_mini(pipeline, maps, frames, save=False):
    cross_attention_maps2 = maps["cross_attention_maps"]
    self_attention_maps = maps["self_attention_maps"]
    was_attention_maps = pipeline.calculate_was_mask(cross_attention_maps2, self_attention_maps,
                                           mask_patch_size=64, use_cross_attention_only=False)
    if save:
        print('was_attention_maps', was_attention_maps.shape)
        print('frames', frames.shape)
        os.makedirs('vis/interpolation', exist_ok=True)
        torch.save(was_attention_maps, 'vis/interpolation/was_attention_maps.pt')
        torch.save(frames, 'vis/interpolation/frames.pt')
    else:
        return was_attention_maps


def _visualize_attention_maps(maps, frames, timestep, indice=0, visualize_cross_attentions=False, dataset_name='default', output_path='output'):
    was, cross = maps["was_attention_maps"], maps["cross_attention_maps"]
    if timestep < 2:
        os.makedirs(f'{output_path}', exist_ok=True)
        post_process = True if 'face' in dataset_name else False
        process_and_save_images(was, frames, dataset_name=dataset_name,
                                output_dir=f'{output_path}/output', 
                                save_frames=True, post_process=post_process)
        visualize_attention_maps(was, output_path=f'{output_path}/was')
        if visualize_cross_attentions:
            visualize_attention_maps(cross, output_path=f'{output_path}/Final_Output/cross_{indice}', interpolation_size=512)
    else:
        os.makedirs(f'{output_path}', exist_ok=True)
        process_and_save_images(was, frames, dataset_name=dataset_name, output_dir=f'{output_path}/output_{indice}')
        if visualize_cross_attentions:
            visualize_attention_maps(cross, output_path=f'{output_path}/cross_{indice}', interpolation_size=512)


def set_inference_timesteps(pipeline, inference_mode, device, **kwargs):
    if inference_mode == 'fast':
        num_inference_steps = 1000
    elif inference_mode == 'modulation':
        num_inference_steps = 50

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    num_inverse_steps = min(num_inference_steps * 2, 999)  # 100
    kwargs['inject_step'] = num_inverse_steps - 20

    pipeline.inverse_scheduler.set_timesteps(num_inverse_steps, device=device)
    inverse_timesteps, num_inverse_steps = pipeline.get_inverse_timesteps(num_inverse_steps, 1, device)
    num_warmup_steps = len(inverse_timesteps) - num_inverse_steps * pipeline.inverse_scheduler.order

    if inference_mode == 'fast':
        # timesteps, inverse_timesteps = timesteps[984:], inverse_timesteps[:-982]
        # timesteps = timesteps*20+1
        # inverse_timesteps = inverse_timesteps*20+1

        # timesteps, inverse_timesteps = timesteps[992:], inverse_timesteps[:-990]
        # timesteps, inverse_timesteps = timesteps[994:], inverse_timesteps[:-992]
        timesteps, inverse_timesteps = timesteps[996:], inverse_timesteps[:-994]
    elif inference_mode == 'modulation':
        timesteps, inverse_timesteps = timesteps[32:], inverse_timesteps[:-30]

    return timesteps, inverse_timesteps, num_warmup_steps, kwargs


def _ddim_inverse(pipeline, latents, prompt_embeds_empty, saved_features, num_inference_steps, 
                  do_classifier_free_guidance, device, STOPTIME, cross_attention_kwargs, extra_step_kwargs,
                  inference_mode='fast', **kwargs):
    timesteps, inverse_timesteps, num_warmup_steps, kwargs = set_inference_timesteps(pipeline, inference_mode, device, **kwargs)

    with pipeline.progress_bar(total=len(inverse_timesteps)) as progress_bar:
        for i, t in enumerate(inverse_timesteps[1:]):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.inverse_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_empty,
                cross_attention_kwargs=cross_attention_kwargs,
                **kwargs,
            ).sample

            if t in timesteps and t > STOPTIME-5:
                # print(f'added time step {t} weights')

                for layer in saved_features.keys():
                    saved_features[layer].append(eval(f"pipeline.unet.{layer}").cpu())

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 1 * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.inverse_scheduler.step(noise_pred, t, latents).prev_sample
            if i == len(inverse_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.inverse_scheduler.order == 0):
                progress_bar.update()
    for layer in saved_features:
        saved_features[layer].reverse()

    return latents, saved_features, timesteps, kwargs


def _process_timestep(pipeline, prompt_embeds, latents, depth_input, t, attention_output_size, 
                      token_ids, average_layers, apply_softmax, video_length, guidance_scale,
                      cross_attention_kwargs, controlnet_keep, controlnet_conditioning_scale, 
                      extra_step_kwargs, return_latent=True , **kwargs):
    do_classifier_free_guidance = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    if depth_input is not None:
        down_block_res_samples, mid_block_res_sample =\
            calculate_controlnet_features(pipeline, depth=depth_input, latent_model_input=latent_model_input, 
                                        prompt_embeds=prompt_embeds, timestep=t, controlnet_keep=controlnet_keep, 
                                        controlnet_conditioning_scale=controlnet_conditioning_scale,)
    else:
        down_block_res_samples, mid_block_res_sample = None, None
    
    noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                  mid_block_additional_residual=mid_block_res_sample, down_block_additional_residuals=down_block_res_samples,
                  cross_attention_kwargs=cross_attention_kwargs, **kwargs,).sample

    _, sd_cross_attention_maps2, sd_self_attention_maps = pipeline.get_attention_map(
                            pipeline.attention_maps, output_size=attention_output_size,
                            token_ids=token_ids, average_layers=average_layers, apply_softmax=apply_softmax,
                            video_length=video_length, guidance_scale=guidance_scale, **kwargs)

    was_attention_maps = pipeline.calculate_was_mask(sd_cross_attention_maps2, sd_self_attention_maps,
                                           mask_patch_size=512, use_cross_attention_only=False)

    was_64_attention_maps = pipeline.calculate_was_mask(sd_cross_attention_maps2, sd_self_attention_maps,
                                           mask_patch_size=64, use_cross_attention_only=False)                                           

    maps = {"cross_attention_maps": sd_cross_attention_maps2, "self_attention_maps": sd_self_attention_maps, 
            "was_attention_maps": was_attention_maps, "was_64_attention_maps": was_64_attention_maps}
    
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    if not return_latent:
        return maps

    step_dict = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
    latents = step_dict.prev_sample

    return latents, maps

def _get_refrence(
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
            saved_features: Dict[str, List[torch.FloatTensor]] = None,
            **kwargs,
    ):
    params = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
        video_length=video_length, frames=frames, height=height, width=width, num_inference_steps=num_inference_steps,
        control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
        generator=generator, eta=eta, guidance_scale=100, **kwargs)

    frames, depth_input, latents, controlnet_keep, extra_step_kwargs, device, do_classifier_free_guidance = (
        params['frames'], params['depth_input'], params['latents'], params['controlnet_keep'], 
        params['extra_step_kwargs'], params['device'], params['do_classifier_free_guidance']
        )

    prompt_embeds_tokens = pipeline._encode_prompt("", device, num_videos_per_prompt, 
        do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=None, 
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
        use_tokens="text_embeds" in training_mode)    

    t = torch.randint(95, 105, [1], dtype=torch.long).to(device=device)
    generate_new_noise = True
    
    with torch.no_grad():
        if generate_new_noise:
            noise = torch.randn_like(latents).to(device=device)
            pipeline.noise = noise
        else:
            noise = pipeline.noise.to(device)
        latents_noisy = pipeline.scheduler.add_noise(latents, noise, t)
        maps = _process_timestep(pipeline, prompt_embeds_tokens, latents_noisy, depth_input=depth_input, 
                                t=t,  attention_output_size=attention_output_size,  token_ids=token_ids, average_layers=average_layers, 
                                apply_softmax=apply_softmax, video_length=video_length, guidance_scale=100, 
                                cross_attention_kwargs=cross_attention_kwargs, controlnet_keep=controlnet_keep, 
                                controlnet_conditioning_scale=controlnet_conditioning_scale, extra_step_kwargs=extra_step_kwargs,
                                return_latent=False, **kwargs)
        for layer in saved_features.keys():
            saved_features[layer].append(eval(f"pipeline.unet.{layer}").cpu())
    
    if kwargs.get("saved_was", None) is not None:
        saved_was = kwargs.get("saved_was")
        maps["was_64_attention_maps"][:saved_was.shape[0]] = saved_was
    
    # visualizing refrence
    # _visualize_attention_maps(maps, frames, timestep=t, indice='refrence', visualize_cross_attentions=False, 
    #                           dataset_name=pipeline.dataset_name, output_path=kwargs.get('output_path', './output'))
    return maps, saved_features

def visualize_ground_truth(pipeline, frames_full, gt_=None, output_path='output'):
    if gt_ is None or len(gt_) == 0:
        return
    gt = []
    for i in range(torch.tensor(gt_).shape[0]):
        gt_mask = torch.tensor(gt_)[i]
        gt_mask = (torch.zeros(pipeline.num_parts, gt_mask.shape[0], gt_mask.shape[1],).to(gt_mask.device)
                .scatter_(0, gt_mask.unsqueeze(0).type(torch.int64), 1.0))
        gt.append(gt_mask)
    gt_mask = torch.stack(gt, dim=0)
    process_and_save_images(gt_mask, frames_full, output_dir=f'{output_path}/GT', save_frames=True, post_process=False)

