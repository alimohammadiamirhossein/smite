import torch
from torch import optim
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    BaseOutput
)


def get_training_mode_conditions():
    crossattn_conditions = {
        "crossattn": {"attn2"},
        "crossattn_kv": {"attn2.to_k", "attn2.to_v"},
        "kv": {"attn2.to_k", "attn2.to_v"},
        "crossattn_q": {"attn2.to_q"},
        "q": {"attn2.to_q"}
    }

    selfattn_conditions = {
        "selfattn": {"attn1"},
        "selfattn_kv": {"attn1.to_k", "attn1.to_v"},
        "kv": {"attn1.to_k", "attn1.to_v"},
        "selfattn_q": {"attn1.to_q"},
        "q": {"attn1.to_q"},
        "allattn": {"attn1", "attn2"},
    }
    
    conditions = {**crossattn_conditions, **selfattn_conditions, "text_embeds":{}}

    combined_modes = {
        f"text_embeds+{key}": {"text_embeds", *values} for key, values in conditions.items() if key != "text_embeds"
    }
    conditions.update(combined_modes)

    return conditions


def configure_optimizers(pipeline, learning_rate: float = 1e-4, training_mode="text_embeds"):
    conditions = get_training_mode_conditions()
    if training_mode not in conditions.keys():
        raise ValueError(f"Invalid training mode: {training_mode}. Valid modes are: {list(conditions.keys())}")

    pipeline.learning_rate = learning_rate
    if training_mode == "text_embeds":
        parameters = [{"params": pipeline.embeddings_to_optimize, "lr": learning_rate}]
        optimizer = getattr(optim, 'Adam')(
            parameters, lr=learning_rate
        )
        pipeline.optimizer = optimizer
    else:
        params_to_optimize = []
        for name, params in pipeline.unet.named_parameters():
            params.requires_grad = False
            if any(cond in name for cond in conditions[training_mode] if cond != "text_embeds"):
                params.requires_grad = True
                params_to_optimize.append({"params": params, "lr": learning_rate})            

        optimizer_unet = getattr(optim, 'Adam')(
            params_to_optimize,
        )
        if "text_embeds" in conditions[training_mode]:
            parameters = [{"params": pipeline.embeddings_to_optimize, "lr": 0.1}]
            optimizer_text_embeds = getattr(optim, 'Adam')(
                parameters,
            )
            pipeline.optimizer = (optimizer_text_embeds, optimizer_unet)
        else:
            pipeline.optimizer = optimizer_unet


def enable_accelerator(pipeline, mixed_precision: Optional[str] = 'fp16', train_dataloader: Optional[torch.utils.data.DataLoader] = None,
                       valid_dataloader: Optional[torch.utils.data.DataLoader] = None):
    if is_accelerate_available():
        prepare_accelerator(pipeline, mixed_precision, train_dataloader, valid_dataloader)
    else:
        raise ImportError("Please install accelerate via `pip install accelerate`")


def prepare_accelerator(pipeline, mixed_precision: str, train_dataloader: Optional[torch.utils.data.DataLoader] = None,
                        valid_dataloader: Optional[torch.utils.data.DataLoader] = None):
    from accelerate import Accelerator
    pipeline.accelerator = Accelerator(mixed_precision=mixed_precision)
    pipeline.train_dataloader = train_dataloader 
    pipeline.valid_dataloader = valid_dataloader
    components = [
        pipeline.vae, pipeline.text_encoder, pipeline.unet, pipeline.controlnet, pipeline.embeddings_to_optimize, 
        pipeline.optimizer, pipeline.train_dataloader, pipeline.valid_dataloader
    ]
    pipeline.vae, pipeline.text_encoder, pipeline.unet, pipeline.controlnet, pipeline.embeddings_to_optimize,\
    pipeline.optimizer, pipeline.train_dataloader, pipeline.valid_dataloader = pipeline.accelerator.prepare(*components)
    initialize_accelerator(pipeline)


def initialize_accelerator(pipeline):
    train_batch_size = 1
    if pipeline.learning_rate is not None:
        pipeline.learning_rate *= train_batch_size * pipeline.accelerator.num_processes
    pipeline.weight_dtype = get_weight_dtype(pipeline)
    move_models_to_device(pipeline)


def get_weight_dtype(pipeline) -> torch.dtype:
    if pipeline.accelerator.mixed_precision == "fp16":
        return torch.float16
    elif pipeline.accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def move_models_to_device(pipeline):
    device = pipeline.accelerator.device
    pipeline.text_encoder.to(device, dtype=pipeline.weight_dtype)
    pipeline.vae.to(device, dtype=pipeline.weight_dtype)

