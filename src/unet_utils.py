import os
import torch
from einops import rearrange
from .optimization import get_training_mode_conditions

def clean_features(pipeline):
    pipeline.attention_maps = {}
    clean_resnet_layers(pipeline.unet)
    clean_attention_layers(pipeline.unet)


def clean_resnet_layers(unet):
    resnet_layers = [
        (x, y) for x in range(4) for y in range(3)
    ]
    for block, resnet in resnet_layers:
        unet.up_blocks[block].resnets[resnet].out_layers_inject_features = None


def clean_attention_layers(unet):
    attention_layers = [
        (i, j, 0) for i in range(1, 4) for j in range(3)
    ]
    for block, attn, trans in attention_layers:
        transformer_block = unet.up_blocks[block].attentions[attn].transformer_blocks[trans]
        transformer_block.attn1.inject_q = None
        transformer_block.attn1.inject_k = None


def inject_out_layers_features(pipeline, layer_name, saved_layer, slice_=None, guidance_scale=7.5):
    if slice_ is not None:
        saved_layer = saved_layer[:, :, slice_]
    eval(f"pipeline.unet.{layer_name}").out_layers_inject_features = saved_layer


def inject_q(pipeline, layer_name, saved_layer, slice_=None, guidance_scale=7.5):
    if slice_ is not None:
        video_lenght = saved_layer.shape[0]//2 if guidance_scale > 1.0 else saved_layer.shape[0]
        query = rearrange(saved_layer, "(b f) d c -> b f d c", f=video_lenght)
        query = query[:, slice_]
        saved_layer = rearrange(query, "b f d c -> (b f) d c")
    eval(f"pipeline.unet.{layer_name}").inject_q = saved_layer


def inject_k(pipeline, layer_name, saved_layer, slice_=None, guidance_scale=7.5):
    if slice_ is not None:
        video_lenght = saved_layer.shape[0]//2 if guidance_scale > 1.0 else saved_layer.shape[0]
        key = rearrange(saved_layer, "(b f) d c -> b f d c", f=video_lenght)
        key = key[:, slice_]
        saved_layer = rearrange(key, "b f d c -> (b f) d c")
    eval(f"pipeline.unet.{layer_name}").inject_k = saved_layer


def get_inject_layer_method(layer_type):
    inject_methods = {
        "out_layers_features": inject_out_layers_features,
        "q": inject_q,
        "k": inject_k,
    }
    return inject_methods.get(layer_type, lambda *args: None)


def inject_layers(pipeline, saved_features, i, slice_=None, guidance_scale=7.5):
    device = pipeline._execution_device
    for layer in saved_features.keys():
        layers = layer.split(".")
        layer_name = ".".join(layers[:-1])
        saved_layer = saved_features[layer][i].to(device)
        inject_layer_method = get_inject_layer_method(layers[-1])
        inject_layer_method(pipeline, layer_name, saved_layer, slice_, guidance_scale)


def get_unet_weights(pipeline, training_mode="text_embeds"):
    conditions = get_training_mode_conditions()
    params_to_save = {}
    for name, params in pipeline.unet.named_parameters():
        if training_mode in conditions:
            if any(cond in name for cond in conditions[training_mode]):
                params_to_save[name] = params
    return params_to_save


def get_embeddings(embeddings_to_optimize):
    params_to_save = {}
    for i, embedding in enumerate(embeddings_to_optimize):
        params_to_save[f"embedding_{i}"] = embedding
    return params_to_save


def save_weights(pipeline, hparams, epoch=0, best=False, saving_path=None):
    training_mode = hparams["training_mode"]
    conditions = get_training_mode_conditions()
    params_to_save = hparams.copy()
    if "text_embeds" in training_mode:
        embed_to_save = get_embeddings(pipeline.embeddings_to_optimize)
        params_to_save["embeddings"] = embed_to_save
    if training_mode in conditions:
        if best is False:
            return
        unet_to_save = get_unet_weights(pipeline, training_mode)
        params_to_save["unet"] = unet_to_save
    torch.save(
        params_to_save, 
        os.path.join(saving_path, f"ckpt_{epoch if not best else 'best'}.pt")
        )


def load_weights(pipeline, loading_path=None):
    params_to_load = torch.load(loading_path)

    pipeline.dataset_name = params_to_load["dataset_name"]
    unet_params = params_to_load["unet"] if "unet" in params_to_load else None
    embeddings = params_to_load["embeddings"] if "embeddings" in params_to_load else None

    if unet_params is not None:
        for name, params in pipeline.unet.named_parameters():
            if name in unet_params:
                # print(name)
                pipeline.unet.state_dict()[name].copy_(unet_params[name])
    
    if embeddings is not None:
        for i in range(len(pipeline.embeddings_to_optimize)):
            pipeline.embeddings_to_optimize[i] = embeddings[f"embedding_{i}"].to(pipeline.embeddings_to_optimize[i].device)
    

