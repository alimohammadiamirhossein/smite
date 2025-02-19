import os
import glob
import PIL
import numpy as np
import cv2
import natsort


def sort_base_size(mask):
    areas = np.array([np.sum(m) for m in mask])
    sort_areas = np.argsort(areas)

    accumulated_mask = np.zeros_like(mask[0])
    for i in sort_areas:
        mask[i] = np.where(accumulated_mask == 1, 0, mask[i])
        accumulated_mask[mask[i] == 1] = 1

    return mask


def process_frame(frame_path):
    masks = natsort.natsorted(glob.glob(os.path.join(frame_path, '*.png')))
    if not os.path.isdir(frame_path):
        return None
    if len(masks) == 0 and len(os.listdir(frame_path)) > 0:
        frame_path = os.path.join(frame_path, os.listdir(frame_path)[0])
        masks = natsort.natsorted(glob.glob(os.path.join(frame_path, '*.png')))

    output_mask = []
    fg_mask = None
    for mask in masks:
        mask_image = PIL.Image.open(mask).convert('L')
        mask_array = np.array(mask_image)
        if fg_mask is None:
            fg_mask = np.zeros_like(mask_array)
        mask_array = np.where(mask_array > 255./2, 255, 0).astype(np.uint8)
        mask_array = (mask_array / 255).astype(np.uint8)
        fg_mask = np.maximum(fg_mask, mask_array)
        output_mask.append(mask_array)

    if len(output_mask) == 0:
        return None

    bg_mask = np.ones_like(fg_mask) - fg_mask
    output_mask = [bg_mask] + output_mask
    output_mask = np.stack(output_mask, axis=0)
    output_mask = sort_base_size(output_mask)
    output_mask = np.argmax(output_mask, axis=0)
    output_mask = cv2.resize(output_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    return output_mask

def read_frame_indices(frame_inds_path):
    with open(frame_inds_path, 'r') as file:
        line = file.readline().strip()  

    frame_inds = [int(x) for x in line.split("[")[1].split("]")[0].split(",") if x.strip()]
    return frame_inds

def extract_gt(data_dir):
    print('gt_extractor.py: Extracting ground truth masks from', data_dir)
    frames = natsort.natsorted(os.listdir(data_dir))
    outputs = []
    for frame in frames:
        frame_path = os.path.join(data_dir, frame)
        output_mask = process_frame(frame_path)
        if output_mask is not None:
            outputs.append(output_mask)
        # if output_mask is not None:
        #     plt.imshow(output_mask)
        #     plt.title(f"Mask for frame: {frame}")
        #     plt.show()
    output_masks = np.stack(outputs, axis=0)

    frame_inds = read_frame_indices(os.path.join(data_dir, 'frames_numbers.txt'))

    return output_masks, frame_inds

