import cv2
import PIL
import torch
import colorsys
import matplotlib
import numpy as np
import torchvision
from PIL import Image
from einops import rearrange
from scipy.signal import savgol_filter
from skimage.morphology import disk, erosion
from concurrent.futures import ThreadPoolExecutor
import decord
decord.bridge.set_bridge('torch')


def get_boundary_and_eroded_mask(mask):
    kernel = disk(3)
    # kernel = disk(1)
    eroded_mask = np.zeros_like(mask)
    boundary_mask = np.zeros_like(mask)
    
    for part_mask_idx in np.unique(mask)[1:]:
        part_mask = np.where(mask == part_mask_idx, 1, 0)
        part_mask_erosion = erosion(part_mask.astype(np.uint8), kernel)
        part_boundary_mask = part_mask - part_mask_erosion
        eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
        boundary_mask = np.where(part_boundary_mask > 0, part_mask_idx, boundary_mask)
        
    return eroded_mask, boundary_mask


def get_colored_segmentation(mask, boundary_mask, image, dataset_name='default'):
    if dataset_name == 'default' or dataset_name == 'face':
        colors = generate_distinct_colors(9)
    elif dataset_name == 'horse':
        colors = generate_distinct_colors(4)
    elif dataset_name == 'car':
        colors = generate_distinct_colors(5)
    elif dataset_name == 'horse2':
        colors = generate_distinct_colors(8)
        # colors[0], colors[2] = colors[2], colors[0]
        # colors[1], colors[3] = colors[3], colors[1]
        # colors[3], colors[4] = colors[4], colors[3]
    elif dataset_name == 'horse3':
        colors = generate_distinct_colors(10 - 1)
        colors[0], colors[5] = colors[5], colors[0]
        colors[1], colors[6] = colors[6], colors[1]
        colors[2], colors[3] = colors[3], colors[2]
        colors[3], colors[8] = colors[8], colors[3]
        colors[0], colors[4] = colors[4], colors[0]
        colors[0], colors[3] = colors[3], colors[0]
        colors[1], colors[2] = colors[2], colors[1]
        colors[3], colors[2] = colors[2], colors[3]
    elif dataset_name == 'face2':
        colors = generate_distinct_colors(7)
        colors[1], colors[4] = colors[4], colors[1]
    elif dataset_name == 'face3':
        colors = generate_distinct_colors(7)
        colors[3], colors[6] = colors[6], colors[3]
        colors[6] = (153, 101, 21)
    elif dataset_name == 'car_horse':
        colors = generate_distinct_colors(11)
        colors[3], colors[7] = colors[7], colors[3]
        colors[6], colors[8] = colors[8], colors[6]
    else:
        colors = generate_distinct_colors(9)
        
    boundary_mask_rgb = 0
    if boundary_mask is not None:
        boundary_mask_rgb = torch.repeat_interleave(boundary_mask[None, ...], 3, 0).type(torch.float)
        for j in range(3):
            for i in range(1, len(colors) + 1):
                boundary_mask_rgb[j] = torch.where(
                    boundary_mask_rgb[j] == i, colors[i - 1][j] / 255, boundary_mask_rgb[j]
                )
    
    mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
    for j in range(3):
        for i in range(1, len(colors) + 1):
            mask_rgb[j] = torch.where(
                mask_rgb[j] == i, colors[i - 1][j] / 255, mask_rgb[j]
            )
    
    if boundary_mask is not None:
        return (boundary_mask_rgb * 0.6 + mask_rgb * 0.4 + image * 0.4).permute(1, 2, 0)
    else:
        return (mask_rgb * 0.6 + image * 0.4).permute(1, 2, 0)


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def visualize_attention_maps(sd_cross_attention_maps, output_path, interpolation_size=None, colorize=True, **kwargs):
    for mm in range(sd_cross_attention_maps.shape[0]):
        if interpolation_size is not None:
            cross_attn_big_size = interpolation_size
            cross_attn_big = torch.zeros(sd_cross_attention_maps[mm].shape[0], cross_attn_big_size, cross_attn_big_size)
            for qq in range(cross_attn_big.shape[0]):
                cross_attn_big[qq] = torch.nn.functional.interpolate(
                    sd_cross_attention_maps[mm, qq][None, None, ...],
                    cross_attn_big_size,
                    mode="bilinear",
                )[0, 0]
        else:
            cross_attn_big = sd_cross_attention_maps[mm]
        
        cross_attn_big = cross_attn_big.detach().cpu().numpy()
        image_array = cross_attn_big.argmax(0)
        
        if colorize:
            image_array = image_array / max(image_array.max(), 1)
            # image_array_clr = colorize_depth_maps(image_array[None, ...], 0., 1., cmap="tab10")
            image_array_clr = colorize_depth_maps(image_array[None, ...], 0., 1., cmap="coolwarm")
            image_array_clr = image_array_clr[0].transpose(1, 2, 0) * 255
            image_array_clr = np.where(image_array[..., None] == 0., 255., image_array_clr)
            image_pil = Image.fromarray(np.uint8(image_array_clr), mode='RGB')
        else:
            image_array = image_array * 255 / max(image_array.max(), 1)
            image_pil = Image.fromarray(np.uint8(image_array), mode='L')
        
        if kwargs.get("writer", None) is not None and mm == 0:
            kwargs["writer"].add_image(kwargs['attention_type'], np.array(image_pil)[None], kwargs['epoch'])
            return
        
        if sd_cross_attention_maps.shape[0] > 1:
            image_pil.save(f"{output_path}_{mm}.png")
        else:
            image_pil.save(f"{output_path}.png")

def create_video_from_images(colored_images, output_dir):
    """Creates a video from a list of images."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = f"{output_dir}_0.mp4"
    
    first_image = np.array(colored_images[0])
    height, width, _ = first_image.shape

    video_writer = cv2.VideoWriter(output_video, fourcc, 15, (width, height))
    for img in colored_images:
        img_np = np.array(img)
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    video_writer.release()

def process_mask_and_frames(final_mask, frames, dataset_name='default', output_dir=None, post_process=False):
    frames, final_mask = frames.cpu(), final_mask.cpu()
    
    if True:
        mask_to_save = torch.nn.functional.interpolate(final_mask, (512, 512), mode="nearest")
        mask_to_save = mask_to_save.argmax(1).to(torch.uint8)
        torch.save(mask_to_save, f"{output_dir}.pt")
        # torch.save(final_mask, f"{output_dir}.pt")
    
    post_process = False
    if post_process and final_mask.shape[0] > 1:
        final_mask = final_mask.permute(1, 0, 2, 3)
        labels = final_mask.argmax(0)
        non_zero_mask = (labels != 0)

        non_zero_mask = non_zero_mask.unsqueeze(0).float()  # Add batch and channel dims, torch.Size([1, 1, 512, 512])
        dilated_non_zero_mask = torch.nn.functional.max_pool2d(non_zero_mask, kernel_size=11, stride=1, padding=5)
        non_zero_mask = dilated_non_zero_mask.squeeze(0).bool() 

        non_zero_mask = non_zero_mask.all(0, keepdim=True).expand_as(labels)
        mean_mask = final_mask.mean(1, keepdim=True)
        mean_mask = mean_mask.expand_as(final_mask)
        motion = final_mask.clone()
        motion[:, non_zero_mask] = final_mask[:, non_zero_mask] - mean_mask[:, non_zero_mask]
        motion_np = motion.cpu().numpy()
        motion_np = savgol_filter(motion_np, 5, 2, axis=1)
        motion = torch.tensor(motion_np, dtype=final_mask.dtype, device=final_mask.device)
        motion[:, non_zero_mask] = motion[:, non_zero_mask] + mean_mask[:, non_zero_mask]
        final_mask[:, non_zero_mask] = motion[:, non_zero_mask]
        final_mask = final_mask.permute(1, 0, 2, 3)
    
    if dataset_name == 'face2':
        final_mask = torchvision.transforms.functional.gaussian_blur(final_mask, kernel_size=7)
    elif dataset_name == 'horse3':
        final_mask = torchvision.transforms.functional.gaussian_blur(final_mask, kernel_size=3)
    else:
        final_mask = torchvision.transforms.functional.gaussian_blur(final_mask, kernel_size=3)
    
    final_mask = final_mask.argmax(1)
    new_mask = final_mask.clone()
    for i in range(final_mask.shape[0]):
        mask_ = final_mask[i].cpu().numpy()
        new_mask[i] = torch.tensor(mask_)
    final_mask = new_mask
    
    return final_mask, frames

def process_and_save_images(final_mask, frames, dataset_name='default', output_dir=None, post_process=False, save_frames=False):
    final_mask, frames = process_mask_and_frames(final_mask, frames, dataset_name, output_dir, post_process)
    
    def process_and_save_image(i):
        eroded_final_mask, final_mask_boundary = get_boundary_and_eroded_mask(
            final_mask[i].cpu()
        )

        frame = (frames[0][:, i] + 1) / 2
        frame = frame.to(torch.float32)

        colored_image = get_colored_segmentation(
            torch.tensor(eroded_final_mask),
            torch.tensor(final_mask_boundary),
            frame.cpu(),
            dataset_name=dataset_name,
        )
        
        frame_image = (frame * 255).cpu().detach().numpy().astype(np.uint8)
        frame_image = PIL.Image.fromarray(frame_image.transpose(1, 2, 0))
        
        colored_image = (colored_image * 255).cpu().detach().numpy().astype(np.uint8)
        colored_image = PIL.Image.fromarray(colored_image)
        
        if output_dir is None:
            colored_image.save(f"outputs/optical_result/colored_image_{i}.png")
            if save_frames:
                frame_image.save(f"outputs/optical_result/frame_image_{i}.png")
        else:
            colored_image.save(f"{output_dir}_{i}.png")
            if save_frames:
                frame_image.save(f"{output_dir}_frame_{i}.png")
        return colored_image

    with ThreadPoolExecutor() as executor:
        colored_images = list(executor.map(process_and_save_image, range(final_mask.shape[0])))
     
    if len(colored_images) > 1:
        create_video_from_images(colored_images, output_dir)
    


def generate_distinct_colors(n):
    colors = []
    if n == 1:
        return [(255, 255, 255)]
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        scaled_rgb = tuple(int(x * 255) for x in rgb)
        colors.append(scaled_rgb)
    return colors

def change_resolution(first_frame, max_size):
    original_height, original_width, _ = first_frame.shape

    if original_width > original_height:
        if original_width > max_size:
            scale = max_size / original_width
            new_width = max_size
            new_height = int(original_height * scale)
        else:
            new_width = original_width
            new_height = original_height
    else:
        if original_height > max_size:
            scale = max_size / original_height
            new_height = max_size
            new_width = int(original_width * scale)
        else:
            new_width = original_width
            new_height = original_height

    return new_height, new_width

def read_video(video_path, video_length, width=512, height=512, frame_rate=None, starting_frame=0):
    if width is None or height is None:
        vr = decord.VideoReader(video_path)
        first_frame = vr[0].numpy()
        height, width, _ = first_frame.shape
        height, width = change_resolution(first_frame, 812)    
    vr = decord.VideoReader(video_path, width=width, height=height)
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(starting_frame, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)
    return video


def get_crops_coords(image_size, patch_size, num_patchs_per_side):
    h, w = image_size
    if num_patchs_per_side == 1:
        x_step_size = y_step_size = 0
    else:
        x_step_size = (w - patch_size) // (num_patchs_per_side - 1)
        y_step_size = (h - patch_size) // (num_patchs_per_side - 1)
    crops_coords = []
    for i in range(num_patchs_per_side):
        for j in range(num_patchs_per_side):
            y_start, y_end, x_start, x_end = (
                i * y_step_size,
                i * y_step_size + patch_size,
                j * x_step_size,
                j * x_step_size + patch_size,
            )
            crops_coords.append([y_start, y_end, x_start, x_end])
    return crops_coords
