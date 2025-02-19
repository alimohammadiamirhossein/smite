import os
import torch
from models.controlnet3d import ControlNet3DModel
from .optimization import get_training_mode_conditions
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def set_embeddings_to_optimize(pipeline, num_of_part, embedding_initialize_prompt=None):
    pipeline.num_parts = num_of_part
    pipeline.part_names = embedding_initialize_prompt.split(" ")
    pipeline.embeddings_to_optimize = []
    if embedding_initialize_prompt is None:
        embedding_initialize_prompt = " ".join(["part" for _ in range(num_of_part)])

    tokenized_prompt = pipeline.tokenizer(
        embedding_initialize_prompt, padding="longest", return_tensors="pt"
    ).input_ids

    device = pipeline.text_encoder.device
    with torch.set_grad_enabled(False):
        text_embedding = pipeline.text_encoder(tokenized_prompt.to(device))[0]

    for i in range(1, num_of_part):
        embedding = text_embedding[:, i : i + 1].clone().detach().to(device)
        embedding.requires_grad_(True)
        pipeline.embeddings_to_optimize.append(embedding)

    pipeline.embeddings_to_optimize = torch.nn.ParameterList(pipeline.embeddings_to_optimize)
    pipeline.unet.to(device)

def encode_prompt(
        pipeline,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        use_tokens: bool = False,
        **kwargs,
):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipeline.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = pipeline.tokenizer.batch_decode(
                untruncated_ids[:, pipeline.tokenizer.model_max_length - 1 : -1]
            )
            logger = kwargs.get("logger", None)
            if logger is not None:
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {pipeline.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = pipeline.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.to(dtype=pipeline.text_encoder.dtype, device=device)

    if use_tokens:
        prompt_embeds = torch.cat([
            prompt_embeds[:, 0:1],
            *list(map(lambda x: x.to(pipeline.device), pipeline.embeddings_to_optimize)),
            prompt_embeds[:, 1 + len(pipeline.embeddings_to_optimize) :],
        ], dim=1)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = pipeline.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None
        
        negative_prompt_embeds = pipeline.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipeline.text_encoder.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds


def align_control_guidance(
    pipeline, control_guidance_start, control_guidance_end, 
):
    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )
    # Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(pipeline.scheduler.timesteps)):
        keeps = [
            1.0 - float(i / len(pipeline.scheduler.timesteps) < s or (i + 1) / len(pipeline.scheduler.timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(pipeline.controlnet, ControlNet3DModel) else keeps)
    return controlnet_keep


def calculate_controlnet_features(
        pipeline,
        depth: torch.FloatTensor,
        latent_model_input: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        timestep: Optional[Union[torch.Tensor, float, int]] = None,
        controlnet_keep: Optional[torch.FloatTensor] = None,
        controlnet_conditioning_scale: Optional[torch.FloatTensor] = None,
):
    control_model_input = latent_model_input.clone()
    controlnet_prompt_embeds = prompt_embeds.clone()
    i = 999 - timestep
    if isinstance(controlnet_keep[i], list):
        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
    else:
        controlnet_cond_scale = controlnet_conditioning_scale
        if isinstance(controlnet_cond_scale, list):
            controlnet_cond_scale = controlnet_cond_scale[0]
        cond_scale = controlnet_cond_scale * controlnet_keep[i]

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        control_model_input,
        timestep,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=depth,
        conditioning_scale=cond_scale,
        return_dict=False,
    )
    return down_block_res_samples, mid_block_res_sample


