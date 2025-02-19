import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from .slicing import create_slices
from .tracking import apply_voting
from .frequency_filter import dct_loss
from .inference import _process_timestep
from utils.image_processing import visualize_attention_maps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def _visualization_during_optimization(was_current, was_tracked, was_refrence, output_path='./output', **kwargs):
    os.makedirs(f'{output_path}/tracking', exist_ok=True)
    visualize_attention_maps(was_current, output_path=f'{output_path}/tracking/was_current', interpolation_size=512)
    visualize_attention_maps(was_tracked, output_path=f'{output_path}/tracking/was_tracked', interpolation_size=512)
    _, was_low_pas = dct_loss(was_refrence, was_tracked, low_pass=True, threshold=kwargs.get("dct_threshold", 0.3))
    visualize_attention_maps(was_low_pas, output_path=f'{output_path}/tracking/was_low_pas', interpolation_size=512)


def _optimize_a_slice(pipeline, slices, s, sliced_latents, sliced_was_tracked, sliced_was_refrence,
                      sliced_depths, saved_features, controlnet_keep, controlnet_conditioning_scale, 
                      extra_step_kwargs, M1, M2, params_to_optimize_a_slice,**kwargs):
    # pipeline.inject_layers(saved_features, i, slice_=slices[s], guidance_scale=guidance_scale)
    slices_dict = {}
    slice_latent = sliced_latents[s]
    slices_dict['latent'] = slice_latent
    if kwargs.get("track_weight", 0.0) > 0.0:
        slice_was_tracked = sliced_was_tracked[s]
        slices_dict['was_tracked'] = slice_was_tracked
    if kwargs.get("regularization_weight", 0.0) > 0.0:
        slice_was_refrence = sliced_was_refrence[s]
        slices_dict['was_refrence'] = slice_was_refrence
    if sliced_depths is not None:
        slice_depths = sliced_depths[s]
    else:
        slice_depths = None
        controlnet_keep = None
        controlnet_conditioning_scale = None
    with torch.set_grad_enabled(True):
        sliced_video_lenght = slices[s].stop - slices[s].start
        new_slice_latent, losses, m1, m2 = _update_latent_with_energy_function(pipeline, 
                                depth_input=slice_depths, latents_dict=slices_dict,
                                controlnet_keep=controlnet_keep, controlnet_conditioning_scale=controlnet_conditioning_scale,
                                video_length=sliced_video_lenght, m1=M1[s], m2=M2[s], 
                                **params_to_optimize_a_slice, **kwargs)

    return new_slice_latent, losses, m1, m2

def optimize_latents(pipeline, 
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
    use_depth = pipeline.controlnet is not None
    
    if use_depth:
        slices = create_slices(latents.shape[2], 2)
    else:
        slices = create_slices(latents.shape[2], 4) # set this based on your gpu memory
    
    M1 = [torch.zeros_like(latents[:,:,slices[0]]) for _ in range(len(slices))]
    M2 = [torch.zeros_like(latents[:,:,slices[0]]) for _ in range(len(slices))]

    pred_track_indices, pred_visibilities = kwargs.get("pred_tracks_indices", None), kwargs.get("pred_visibilities", None)
    was_tracked = apply_voting(was_current, pred_track_indices, pred_visibilities)
    _visualization_during_optimization(was_current, was_tracked, was_refrence, output_path=kwargs.get('output_path', './output'))
    
    losses_to_track = {'counter': 0, 'total_loss': 0.0, 'loss_tracking': 0.0, 'loss_refrence': 0.0}
    
    with tqdm(total=15, desc="Optimizing Latents", leave=True) as pbar:
        for iter_ in range(15):
            sliced_latents = [latents[:, :, s, ...] for s in slices]
            
            sliced_was_tracked = (
                [was_tracked[s, ...] for s in slices]
                if kwargs.get("track_weight", 0.0) > 0.0 else None
            )
            
            sliced_was_refrence = (
                [was_refrence[s, ...] for s in slices]
                if kwargs.get("regularization_weight", 0.0) > 0.0 else None
            )
            
            sliced_depths = (
                [depth_input[:, :, s, ...] for s in slices] if use_depth else None
            )
        
            total_loss = 0
            pipeline.clean_features()

            for s in range(len(slices)):
                params_to_optimize_a_slice = {'prompt_embeds': prompt_embeds, 't': t, 'guidance_scale': guidance_scale,
                                      'negative_prompt_embeds': None, 'attention_output_size': attention_output_size,
                                      'token_ids': token_ids, 'average_layers': average_layers, 'apply_softmax': apply_softmax,
                                      'cross_attention_kwargs': cross_attention_kwargs, 'extra_step_kwargs': extra_step_kwargs,
                                      'iter_': iter_}

                new_slice_latent, losses, m1, m2 = _optimize_a_slice(pipeline, slices=slices, 
                                                                     s=s, sliced_latents=sliced_latents, 
                                                                     sliced_was_tracked=sliced_was_tracked,
                                                                     sliced_was_refrence=sliced_was_refrence,
                                                                     sliced_depths=sliced_depths,
                                                                     saved_features=saved_features, 
                                                                     controlnet_keep=controlnet_keep, 
                                                                     controlnet_conditioning_scale=controlnet_conditioning_scale, 
                                                                     extra_step_kwargs=extra_step_kwargs, 
                                                                     M1=M1, M2=M2, 
                                                                     params_to_optimize_a_slice=params_to_optimize_a_slice,
                                                                     **kwargs)
                M1[s] = m1
                M2[s] = m2
                sliced_latents[s] = new_slice_latent
                torch.cuda.empty_cache()
                for key in losses_to_track:
                    if key == 'total_loss':
                        losses_to_track[key] = (losses_to_track['counter']-1)*losses_to_track[key] + losses[key]
                        losses_to_track[key] /= (losses_to_track['counter']+1)
                    elif key != 'counter':
                        losses_to_track[key] = losses[key]
                losses_to_track['counter'] += 1
            pbar.set_postfix({k:v for k,v in losses_to_track.items() if k != 'counter'})
            pbar.update()

            latents = torch.cat(sliced_latents, dim=2)
    return latents

def _compute_loss(pipeline, latents_dict, was_dst, **kwargs):
    loss, loss_refrence, loss_tracking = 0.0, torch.tensor(0.0), torch.tensor(0.0)
    if kwargs.get("regularization_weight", 0.0) > 0.0:
        was_refrence = latents_dict['was_refrence']
        loss_refrence = _compute_refrence_loss(was_dst, was_refrence, **kwargs)
        loss += kwargs.get("regularization_weight", 0.0) * loss_refrence

    if kwargs.get("track_weight", 0.0) > 0.0: #and kwargs.get("i", 0) > 0:
        was_tracked = latents_dict['was_tracked'].clone()
        loss_tracking = _compute_tracking_loss(was_dst, was_tracked, pipeline, **kwargs)
        loss += kwargs.get("track_weight", 0.0) * loss_tracking
        
    return loss, loss_refrence, loss_tracking

def _compute_refrence_loss(was_dst, was_refrence, **kwargs):
    threshold = 0.999 if kwargs.get("i", 0) == 0 else kwargs.get("dct_threshold", 0.0)
    loss_refrence_based_optimization, _ = dct_loss(was_dst.float(), was_refrence, low_pass=True, threshold=threshold)
    # loss_refrence_based_optimization = F.mse_loss(was_dst.float(), was_refrence) * pipeline.num_parts
    return loss_refrence_based_optimization 

def _compute_tracking_loss(was_dst, was_tracked, pipeline, **kwargs):
    # loss_high_was, _ = dct_loss(was_dst.float(), was_tracked, low_pass=False, threshold=dct_threshold)
    # loss += kwargs.get("track_weight", 0.0) * loss_high_was * pipeline.num_parts
    # loss_tracking_based_optimization = F.mse_loss(was_dst.float(), was_tracked) * pipeline.num_parts
    pixel_weights = _calculate_pixel_weights(pipeline, was_tracked)
    loss_tracking_based_optimization = F.cross_entropy(was_dst.float(), was_tracked.argmax(dim=1), weight=pixel_weights)
    return loss_tracking_based_optimization

def _calculate_pixel_weights(pipeline, was_tracked):
    mask = was_tracked.argmax(dim=1)
    values, counts = torch.unique(mask, return_counts=True)
    num_pixels = torch.zeros(pipeline.num_parts, dtype=torch.int64).to(pipeline.device)
    num_pixels[values.type(torch.int64)] = counts.type(torch.int64)
    num_pixels[0] = 0
    pixel_weights = torch.where(num_pixels > 0, num_pixels.sum() / (num_pixels + 1e-6), 0)
    pixel_weights[0] = 1
    return pixel_weights

def _update_latent_with_energy_function(
    pipeline,
    prompt_embeds: torch.FloatTensor,
    latents_dict: Dict[str, torch.FloatTensor],
    t: int,
    depth_input: Optional[torch.FloatTensor] = None,
    video_length: Optional[int] = 1,
    guidance_scale: float = 7.5,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    attention_output_size: Optional[int] = 256,
    token_ids: Union[List[int], Tuple[int]] = None,
    average_layers: Optional[bool] = True,
    apply_softmax: Optional[bool] = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_keep: Optional[torch.FloatTensor] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    extra_step_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
    ):
    latents = latents_dict['latent'].requires_grad_(True)
    latents_new, maps = _process_timestep(pipeline, prompt_embeds, latents, depth_input=depth_input, 
                                t=t,  attention_output_size=attention_output_size,  token_ids=token_ids, average_layers=average_layers, 
                                apply_softmax=apply_softmax, video_length=video_length, guidance_scale=guidance_scale, 
                                cross_attention_kwargs=cross_attention_kwargs, controlnet_keep=controlnet_keep, 
                                controlnet_conditioning_scale=controlnet_conditioning_scale, extra_step_kwargs=extra_step_kwargs,
                                **kwargs)

    cross_a, self_a, was_dst = maps["cross_attention_maps"], maps["self_attention_maps"],  maps["was_64_attention_maps"]    
    
    loss, loss_refrence, loss_tracking = _compute_loss(pipeline, latents_dict, was_dst, **kwargs)
    losses = {'total_loss': loss.item(), 'loss_tracking': loss_tracking.item(), 'loss_refrence': loss_refrence.item()}
    del latents_new, maps, cross_a, self_a, was_dst, loss_refrence, loss_tracking

    latents, M1, M2 = _update_latent_with_gradient(pipeline, latents, loss, t, **kwargs)

    return latents, losses, M1, M2

def _update_latent_with_gradient(pipeline, latents, loss, t, **kwargs):
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
    sigma = (1 - pipeline.scheduler.alphas[t]) / pipeline.scheduler.alphas[t]
    lr = 10
    
    M1, M2, M1_hat, M2_hat = _update_moments(grad_cond, **kwargs)

    velocity = lr * sigma * M1_hat / (M2_hat.sqrt() + 1e-5)
    latents = latents - velocity
    latents = latents.detach()
    del grad_cond, velocity, M1_hat, M2_hat
    return latents, M1, M2

def _update_moments(grad_cond, **kwargs):
    M1 = kwargs.get("m1", 0.)
    M2 = kwargs.get("m2", 0.)
    b1, b2 = 0.9, 0.999

    M1 = b1 * M1 + (1 - b1) * grad_cond
    M2 = b2 * M2 + (1 - b2) * grad_cond**2
    iter_ = kwargs.get("iter_", 0)

    M1_hat = M1 / (1 - b1**(iter_ + 1))
    M2_hat = M2 / (1 - b2**(iter_ + 1))

    return M1, M2, M1_hat, M2_hat
