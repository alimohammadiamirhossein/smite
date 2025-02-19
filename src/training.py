import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torchvision
import torch.nn.functional as F
from utils.image_processing import visualize_attention_maps, process_and_save_images
from .embeddings import align_control_guidance, calculate_controlnet_features
from .unet_utils import save_weights
from .metrics import calculate_iou
from .was_attention import calculate_was_mask
from .frequency_filter import frequency_loss
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def train_pipeline(
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
    train_loss = 0.0
    best_iou = 0.0
    valid_epochs_interval = 1
    
    if pipeline.accelerator is not None:
        weight_dtype = torch.float32
        if pipeline.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif pipeline.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float16

    hparams = {
        "dataset_name": pipeline.dataset_name,
        "dataset_size": len(pipeline.train_dataloader.dataset),
        "guidance_scale": guidance_scale,
        "num_of_epochs": num_of_epochs,
        "learning_rate": pipeline.learning_rate,
        "time_interval": "-".join(str(x) for x in kwargs.get("time_interval", [5, 200])),
        "use_controlnet": pipeline.controlnet is not None,
        "training_mode": kwargs.get("training_mode", "text_embeds"),
        "coef_loss_sd": kwargs.get("coef_loss_sd", 0.0),
        "coef_loss_fourier": kwargs.get("coef_loss_fourier", 0.0),
        }

    progress_bar = tqdm(total=num_of_epochs, desc='Training')
    pipeline.unet.train()
    for epoch in range(num_of_epochs):
        losses = training_step(pipeline, prompt=prompt, mode='train', video_length=video_length,
        height=height, width=width, guidance_scale=guidance_scale, epoch=epoch,
        attention_output_size=attention_output_size, token_ids=token_ids, average_layers=average_layers, 
        apply_softmax=apply_softmax, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, 
        eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, output_type=output_type, return_dict=return_dict, 
        callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, 
        controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start, 
        control_guidance_end=control_guidance_end, generate_new_noise=generate_new_noise, weight_dtype=weight_dtype,
        **kwargs,
        )
        loss, loss1, loss2, loss_sd, loss_freq = losses["total"], losses["loss1"], losses["loss2"], losses["loss_sd"], losses["loss_freq"]
        train_loss += (loss - train_loss) / (epoch + 1)
        pipeline.writer.add_scalar("Loss/train", train_loss, epoch)
        pipeline.writer.add_scalar("Loss/loss1", loss1, epoch)
        pipeline.writer.add_scalar("Loss/loss2", loss2, epoch)
        pipeline.writer.add_scalar("Loss/loss_sd", loss_sd, epoch)
        pipeline.writer.add_scalar("Loss/loss_freq", loss_freq, epoch)

        if epoch % valid_epochs_interval == 0:
            valid_output = training_step(pipeline, prompt=prompt, mode='valid', video_length=video_length,
            height=height, width=width, guidance_scale=guidance_scale, epoch=epoch,
            attention_output_size=attention_output_size, token_ids=token_ids, average_layers=average_layers, 
            apply_softmax=apply_softmax, negative_prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, 
            eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, output_type=output_type, return_dict=return_dict, 
            callback=callback, callback_steps=callback_steps, cross_attention_kwargs=cross_attention_kwargs, 
            controlnet_conditioning_scale=controlnet_conditioning_scale, control_guidance_start=control_guidance_start, 
            control_guidance_end=control_guidance_end, generate_new_noise=generate_new_noise, weight_dtype=weight_dtype,
            complete_dataset=False, #True if epoch % 20 == 19 else False,
            **kwargs,
            )
            ious = []
            iou_str = ""
            for part_name in pipeline.part_names:
                iou = valid_output[f"iou_{part_name}"]
                pipeline.writer.add_scalar(f"IoU/{part_name}", iou, epoch)
                ious.append(iou)
                iou_str += f"{part_name}: {iou:.4f}, "
            pipeline.writer.add_scalar("IoU/mean", np.mean(ious), epoch)
            iou_str += f"Mean: {np.mean(ious):.4f}"
            
            if np.mean(ious) > best_iou:
                print(f"Better IoU at Epoch: {epoch}, {iou_str}")
                best_iou = np.mean(ious)
                hparams["m_iou"] = best_iou
                save_weights(pipeline, hparams=hparams, epoch=epoch, best=True, saving_path=f"{pipeline.writer_path}")
            # if epoch % 20 == 0:
            #     print(f"Epoch: {epoch}, {iou_str}")
        else:
            ious = [0.0 for _ in pipeline.part_names]
    
        progress_bar.set_postfix_str(
            f"total loss: {train_loss:.4f}, loss1: {loss1:.4f}, loss_sd: {loss_sd:.4f}, loss2: {loss2:.4f}, "
            f"loss_freq: {loss_freq:.4f}, Mean IoU: {np.mean(ious):.4f}"
        )
        progress_bar.update()
        if epoch % 50 == 0 or (30 < epoch < 250 and epoch % 5 == 0):
            save_weights(pipeline, hparams=hparams, epoch=epoch, best=False, saving_path=f"{pipeline.writer_path}")

    if "m_iou" in hparams:
        del hparams["m_iou"]
    pipeline.writer.add_hparams(hparams, {'m_iou': float(best_iou)}, run_name=pipeline.writer_path)

def compute_loss(cross_attention, was_attention, cross_gt, was_gt, noise, noise_gt, pixel_weights, num_parts, **kwargs):
    loss1 = F.cross_entropy(
        cross_attention[None, ...].float(),
        cross_gt.type(torch.long),
        weight=pixel_weights,
    )
    loss2 = F.mse_loss(was_attention.float(), was_gt) * num_parts

    loss_freq = frequency_loss(was_attention.float(), was_gt)

    loss_sd = F.mse_loss(noise.float(), noise_gt.float(), reduction="none").mean([1, 2, 3]).mean()

    # calculate the total loss
    coef_loss_sd = kwargs.get("coef_loss_sd", 0.0)
    epoch = kwargs.get("epoch", 0)
    coef_loss_fourier = kwargs.get("coef_loss_fourier", 0.0) if epoch > 5 else 0.0
    loss = loss1 + loss_sd * coef_loss_sd + loss2 + loss_freq * coef_loss_fourier
    losses = {"total": loss, "loss1": loss1, "loss2": loss2, "loss_sd": loss_sd, "loss_freq": loss_freq}

    return losses

def backward_and_optimize(loss, pipeline, epoch=0, scale=1., training_mode="text_embeds"):
    if pipeline.accelerator is not None:
        avg_loss = pipeline.accelerator.gather(loss.repeat(1)).mean()
        pipeline.accelerator.backward(loss * scale)
        if pipeline.accelerator.sync_gradients:
            max_grad_norm = 1000.0
            pipeline.accelerator.clip_grad_norm_(pipeline.embeddings_to_optimize, max_grad_norm)
    else:
        loss.backward()
    
    if "text_embeds+" in training_mode:
        if epoch <= 50 or (epoch % 10 == 0):
            pipeline.optimizer[1].zero_grad()
            pipeline.optimizer[0].step()
            pipeline.optimizer[0].zero_grad()
        if epoch > 50 and (epoch % 10 != 0):
            for embed in pipeline.embeddings_to_optimize:
                embed.requires_grad = False
            pipeline.optimizer[0].zero_grad()
            pipeline.optimizer[1].step()
            pipeline.optimizer[1].zero_grad()
    else:
        if "text_embeds" not in training_mode:
            if epoch < 10:
                for param_group in pipeline.optimizer.param_groups:
                    param_group["lr"] = pipeline.learning_rate * 100
            elif epoch < 40:
                for param_group in pipeline.optimizer.param_groups:
                    param_group["lr"] = pipeline.learning_rate * 10
            else:
                for param_group in pipeline.optimizer.param_groups:
                    param_group["lr"] = pipeline.learning_rate
        pipeline.optimizer.step()
        pipeline.optimizer.zero_grad()


def training_step(
    pipeline,
    prompt: Union[str, List[str]] = None,
    mode: Optional[str] = "train",
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
    is_train = mode == "train"
    os.makedirs(f'vis/{mode}', exist_ok=True)
    t2i_transform = torchvision.transforms.ToPILImage()
    dataloader = pipeline.train_dataloader if is_train else pipeline.valid_dataloader
    if is_train:
        pipeline.unet.train()
    else:
        pipeline.unet.eval()

    if is_train:
        output = {
                "total": np.array([]), "loss1": np.array([]), "loss2": np.array([]), "loss_sd": np.array([]),
                "loss_freq": np.array([])
            }
    else:
        output = {f'iou_{part_name}':[] for part_name in pipeline.part_names}

    for step, batch in enumerate(dataloader):
        if pipeline.dataset_name == 'face':
            valid_condition = (step <= 10)
        else:
            valid_condition = True
            
        if not is_train and not complete_dataset and not valid_condition:
            continue
        visualize = torch.rand(1).item() < 0.1
        depth = batch.get("depth", None)
        use_depth = (depth is not None) and (pipeline.controlnet is not None)
        image, mask, prompt = batch["image"], batch["mask"], batch["prompt"]
        prompt = prompt[0]
        image = image[0, 0]#[[2, 1, 0]]
        image_to_save = image.clone()
        image = t2i_transform((image*255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        if visualize:
            image.save(f"vis/{mode}/input.png")
        frames, masks = [image], [mask[0, 0]]
        depths = [depth[0].cpu().numpy()] if use_depth else None
        prompt_embeds = None
        
        params = pipeline.check_and_initialize(prompt=prompt, callback_steps=callback_steps, negative_prompt=negative_prompt, 
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_videos_per_prompt=num_videos_per_prompt,
            video_length=video_length, frames=frames, height=height, width=width,
            control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, 
            generator=generator, eta=eta, guidance_scale=guidance_scale, depths=depths, save_path=f"vis/{mode}")

        frames, depths, latents, controlnet_keep, extra_step_kwargs, device, do_classifier_free_guidance = (
            params['frames'], params['depth_input'], params['latents'], params['controlnet_keep'], 
            params['extra_step_kwargs'], params['device'], params['do_classifier_free_guidance']
            )
        
        if epoch % 50 == 0:
            pipeline.writer.add_image("input", image_to_save, epoch)
            if use_depth:
                pipeline.writer.add_image("depth", depths[0,:,0], epoch)
        
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

        if use_depth:
            if epoch % 50 == 0:
                pipeline.writer.add_image("depth", depths[0,:,0], epoch)

        t = kwargs.get("time_interval", [5, 200])
        if mode == "valid":
            t = [95, 105]
        if len(t) == 2:
            t = torch.randint(t[0], t[1] + 1, [1], dtype=torch.long)
        t = t.to(device=device)

        with torch.enable_grad() if is_train else torch.no_grad():
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

        if visualize:
            visualize_attention_maps(
                sd_cross_attention_maps2,
                output_path=f"./vis/{mode}/inference_cross_attention_maps2",
                interpolation_size=512,
                attention_type="cross_attention_maps",
                epoch=epoch,
                writer=pipeline.writer if epoch % 50 == 0 else None,
                colorize=False,
            )

        was_attention_maps = \
            pipeline.calculate_was_mask(sd_cross_attention_maps2.clone(), sd_self_attention_maps.clone(),
                                                mask_patch_size=512, use_cross_attention_only=False)

        if visualize:
            visualize_attention_maps(was_attention_maps, output_path=f"./vis/{mode}/inference_was_attention_maps2",
                attention_type="was_attention_maps", writer=pipeline.writer if epoch % 50 == 0 else None,
                epoch=epoch, colorize=False)
            process_and_save_images(was_attention_maps, frames, dataset_name=pipeline.dataset_name,
                                    output_dir=f'vis/{mode}/output', save_frames=False)

        sd_self_attention_maps = sd_self_attention_maps[0]
        pipeline.attention_maps = {}

        image, mask = frames[:, :, 0], masks[0].unsqueeze(0)
        # mask = F.interpolate(mask.unsqueeze(0), size=attention_output_size, mode="bilinear")[0]
        num_pixels = torch.zeros(pipeline.num_parts, dtype=torch.int64).to(pipeline.device)
        values, counts = torch.unique(mask, return_counts=True)
        num_pixels[values.type(torch.int64)] = counts.type(torch.int64)
        num_pixels[0] = 0
        pixel_weights = torch.where(num_pixels > 0, num_pixels.sum() / (num_pixels + 1e-6), 0)
        pixel_weights[0] = 1
        mask = mask[0]
        sd_cross_attention_maps2 = sd_cross_attention_maps2[0]

        sd_cross_attention_maps2_softmax = sd_cross_attention_maps2.softmax(dim=0)
        small_sd_cross_attention_maps2 = F.interpolate(sd_cross_attention_maps2_softmax[None, ...], 
                                                                            64, mode="bilinear")[0]
        was_attention_map = (sd_self_attention_maps[None, ...]*
            small_sd_cross_attention_maps2.flatten(1, 2)[..., None, None]).sum(dim=1)

        one_shot_mask = (
            torch.zeros(pipeline.num_parts, mask.shape[0], mask.shape[1],).to(mask.device)
                .scatter_(0, mask.unsqueeze(0).type(torch.int64), 1.0))

        if visualize:
            visualize_attention_maps(one_shot_mask.unsqueeze(0), output_path=f"./vis/{mode}/gt_mask",
                interpolation_size=512, attention_type="gt_mask", epoch=epoch,
                writer=pipeline.writer if epoch % 50 == 0 else None, colorize=False)

        if is_train:
            losses = compute_loss(cross_attention=sd_cross_attention_maps2, was_attention=was_attention_map, 
                                cross_gt=mask[None, ...], was_gt=one_shot_mask, pixel_weights=pixel_weights,
                                noise=noise_pred, noise_gt=noise, epoch=epoch, num_parts=pipeline.num_parts, 
                                **kwargs)
            
            backward_and_optimize(losses['total'], pipeline, epoch=epoch, scale=0.01, training_mode=training_mode)

            for key in losses:
                output[key] = np.append(output[key], losses[key].item())
        
        else:
            was_attention = calculate_was_mask(pipeline, sd_cross_attention_maps2.unsqueeze(0),
                         sd_self_attention_maps.unsqueeze(0), mask_patch_size=512, use_cross_attention_only=False)[0]
            pred_mask = was_attention.argmax(0)
            for ind, part_name in enumerate(pipeline.part_names):
                gt = torch.where(mask == ind, 1, 0).type(torch.int64)
                pred = torch.where(pred_mask == ind, 1, 0).type(torch.int64)
                if torch.all(gt == 0):
                    continue
                iou = calculate_iou(pred, gt)
                output[f"iou_{part_name}"] = np.append(output[f"iou_{part_name}"], iou.item())

    for key in output:
        output[key] = output[key].mean() if len(output[key]) > 0 else 0

    return output

