import os
import random
import numpy as np
import torch, torchvision
from .transfer_weights import transfer_unets
from .image_processing import read_video
from models import UNet3DConditionModel, ControlNet3DModel
from src import SMITEPipeline
from utils import SMITEDataset
from utils.image_processing import get_crops_coords
from utils.gt_extractor import extract_gt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, DDIMInverseScheduler, UNet2DConditionModel, ControlNetModel


def setup_output_directory(path):
    os.makedirs(path, exist_ok=True)


def adjust_dimensions(args):
    args.height = (args.height // 32) * 32
    args.width = (args.width // 32) * 32


def load_tokenizer_and_text_encoder(model_id, device):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device=device)
    return tokenizer, text_encoder


def load_vae(model_id, device):
    return AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device=device)


def load_unets(model_id, device, dtype=torch.float32):
    dtype = torch.float32
    unet3d = UNet3DConditionModel.from_pretrained_2d(None, subfolder="unet").to(device=device, dtype=dtype)
    unet2d = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    unet = transfer_unets(unet3d, unet2d)
    del unet2d, unet3d
    return unet


def load_controlnet(device, dtype=torch.float32):
    controlnet2d = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-depth-diffusers", torch_dtype=dtype).to(device=device)
    controlnet3d = ControlNet3DModel.from_pretrained_2d(None, subfolder="unet").to(device=device, dtype=dtype)
    controlnet = transfer_unets(controlnet3d, controlnet2d, is_controlnet=True)
    del controlnet2d, controlnet3d
    return controlnet


def load_schedulers(model_id):
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    inverse = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
    return scheduler, inverse


def set_generator(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def load_pipeline(model_id, device, attention_layers_to_use, use_controlnet, dtype=torch.float32):
    tokenizer, text_encoder = load_tokenizer_and_text_encoder(model_id, device)
    vae = load_vae(model_id, device)
    unet = load_unets(model_id, device, dtype=dtype)

    controlnet = load_controlnet(device, dtype=dtype) if use_controlnet else None

    scheduler, inverse = load_schedulers(model_id)
    pipe = SMITEPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                          controlnet=controlnet, scheduler=scheduler, inverse_scheduler=inverse)
    pipe.set_attention_hook(attention_layers_to_use)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def setup_training_dataset(input_prompt, train_dir, min_crop_ratio, batch_size=1, flip_dataset=True, 
                           dataset_name="pascal", is_train=True):
    num_parts = len(input_prompt.split(" "))
    mask_size = 64 if is_train else 512
    dataset = SMITEDataset(train_dir, num_frames=1, train=is_train, num_parts=num_parts,
                                     min_crop_ratio=min_crop_ratio, flip=flip_dataset, prompt=input_prompt,
                                     mask_size=mask_size, dataset_name=dataset_name)
    shuffle = is_train
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def setup_training_pipeline(pipeline, train_dir, val_dir, input_prompt, min_crop_ratio, dataset_name, flip_dataset=True,
                            batch_size=1, training_mode='text_embeds', learning_rate=0.1, dtype=torch.float32):
    pipeline._initialize_logger(dataset_name)
    num_parts = len(input_prompt.split(" "))
    train_dataloader = setup_training_dataset(input_prompt, train_dir, min_crop_ratio, batch_size, flip_dataset=flip_dataset, 
                                              dataset_name=dataset_name, is_train=True)
    val_dataloader = setup_training_dataset(input_prompt, val_dir, min_crop_ratio, batch_size, flip_dataset=False, 
                                            dataset_name=dataset_name, is_train=False)
    
    pipeline.set_embeddings_to_optimize(num_parts, input_prompt)
    pipeline.configure_optimizers(learning_rate=learning_rate, training_mode=training_mode)
    
    pipeline.enable_accelerator(train_dataloader=train_dataloader, valid_dataloader=val_dataloader, 
                                mixed_precision='fp16' if dtype == torch.float16 else 'no')


def setup_validation_pipeline(pipeline, train_dir, val_dir, input_prompt, min_crop_ratio, dataset_name,
                            batch_size=1, training_mode='text_embeds', ckpt_path=None):
    num_parts = len(input_prompt.split(" "))
    train_dataloader = setup_training_dataset(input_prompt, train_dir, min_crop_ratio, batch_size, dataset_name, is_train=True)
    val_dataloader = setup_training_dataset(input_prompt, val_dir, min_crop_ratio, batch_size, dataset_name, is_train=False)
    
    pipeline.set_embeddings_to_optimize(num_parts, input_prompt)
    if ckpt_path is not None:
        pipeline.load_weights(ckpt_path)
    
    pipeline.enable_accelerator(train_dataloader=train_dataloader, valid_dataloader=val_dataloader, mixed_precision='fp16')


def setup_inference_pipeline(pipeline, prompt, ckpt_path=None):
    pipeline.initialize_tracker()
    num_parts = len(prompt.split(" "))
    pipeline.set_embeddings_to_optimize(num_parts, prompt)
    if ckpt_path is not None:
        pipeline.load_weights(ckpt_path)
    pipeline.enable_accelerator(mixed_precision='fp16')

def get_crop_coords_if_needed(args, patch_size=450):
    if "face" in args.dataset_name:
        crop_coords = get_crops_coords([512, 512], patch_size=patch_size, num_patchs_per_side=2)
        return crop_coords
    return None

def _read_and_transform_video(video_path, video_length, width, height, frame_rate, starting_frame):
    t2i_transform = torchvision.transforms.ToPILImage()
    video = read_video(video_path=video_path, video_length=video_length, width=width, height=height, frame_rate=frame_rate, starting_frame=starting_frame)
    transformed_frames = [t2i_transform(((frame + 1) / 2 * 255).to(torch.uint8)) for frame in video]
    return transformed_frames


def _get_depth_video_path(video_path):
    fname = video_path.split("/")[-1].split(".")[0]
    depth_path = '/'.join(video_path.split("/")[:-1]) + f"/{fname}_depth_bw.mp4"
    return depth_path

def get_frame_inds(starting_frame, frame_rate, num_frames):
    return [starting_frame + i * frame_rate for i in range(num_frames)]

def _process_ground_truth(gt_path, frame_inds):
    real_gt, gt_inds = extract_gt(gt_path)
    max_gt_inds = max(gt_inds)
    candidates = sorted([x for x in frame_inds if x > max_gt_inds])

    new_video_length = None
    if len(candidates) > 3:
        ind_tmp = 3
        while ind_tmp >= 0:
            candidate = candidates[min(ind_tmp, len(candidates) - 1)]
            frame_inds = [x for x in frame_inds if x <= candidate]
            if len(frame_inds) % 4 == 0:
                new_video_length = len(frame_inds)
                break
            ind_tmp -= 1
    output_dict = {"gt": real_gt, "gt_inds": gt_inds, "frame_inds": frame_inds}
    if new_video_length is not None:
        output_dict["new_video_length"]=new_video_length
        
    return output_dict

def load_video(video_path, video_length, width, height, frame_rate, use_controlnet, gt_path=None, starting_frame=0, hierarchical_segmentation=False):
    if hierarchical_segmentation:
        height, width = None, None
    real_frames = _read_and_transform_video(video_path, video_length, width, height, frame_rate, starting_frame)
    video_length = len(real_frames)
    frame_inds = get_frame_inds(starting_frame, frame_rate, video_length)
    
    real_depths = []
    if use_controlnet:
        depth_path = _get_depth_video_path(video_path)
        real_depths = _read_and_transform_video(depth_path, video_length, width, height, frame_rate, starting_frame)

    if gt_path and len(gt_path) > 0:
        gt_dict = _process_ground_truth(gt_path, frame_inds)
        new_video_length = gt_dict.get('new_video_length')
        
        if new_video_length:
            real_frames = real_frames[:new_video_length]
            real_depths = real_depths[:new_video_length]
    else:
        gt_dict = {}
        new_video_length = None

    return real_frames, real_depths, gt_dict



