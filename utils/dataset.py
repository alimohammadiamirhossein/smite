import os
import cv2
import torch
import random
import natsort
import numpy as np
from PIL import Image
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as TorchDataset


def get_random_crop_coordinates(crop_scale_range, image_width, image_height):
    rand_number = random.random()
    rand_number *= crop_scale_range[1] - crop_scale_range[0]
    rand_number += crop_scale_range[0]
    patch_size = int(rand_number * min(image_width, image_height))
    if patch_size != min(image_width, image_height):
        x_start = random.randint(0, image_width - patch_size)
        y_start = random.randint(0, image_height - patch_size)
    else:
        x_start = 0
        y_start = 0
    return x_start, x_start + patch_size, y_start, y_start + patch_size


def resize_mask_to_image(image, mask):
    h_img, w_img = image.shape[:2]
    resized_mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    return resized_mask

def crop_image_by_mask(image, mask, expand_ratio=0.9):
    y_indices, x_indices = np.where(mask == 1)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("No region with mask value 1 found.")
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    x_min = max(0, int(x_min - box_width * expand_ratio))
    x_max = min(image.shape[1], int(x_max + box_width * expand_ratio))
    y_min = max(0, int(y_min - box_height * expand_ratio))
    y_max = min(image.shape[0], int(y_max + box_height * expand_ratio))
    
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_mask

def zoom_out(image, mask=None, depth=None, min_scale=0.2, max_scale=0.5):
    h, w = image.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_h, new_w = int(h * scale), int(w * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = None
    if depth is not None:
        if len(depth.shape) == 3:
            resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_depth = None

    canvas_image = np.zeros_like(image)
    canvas_mask = np.zeros_like(mask) if mask is not None else None
    canvas_depth = np.zeros_like(depth) if depth is not None else None

    max_x = w - new_w
    max_y = h - new_h
    x_start = random.randint(0, max_x) if max_x > 0 else 0
    y_start = random.randint(0, max_y) if max_y > 0 else 0

    canvas_image[y_start:y_start+new_h, x_start:x_start+new_w] = resized_image
    if mask is not None:
        canvas_mask[y_start:y_start+new_h, x_start:x_start+new_w] = resized_mask
    if depth is not None:
        canvas_depth[y_start:y_start+new_h, x_start:x_start+new_w] = resized_depth

    return canvas_image, canvas_mask, canvas_depth

class SMITEDataset(TorchDataset):
    def __init__(
            self,
            data_dir,
            train=True,
            num_frames=12,
            mask_size=512,
            num_parts=1,
            min_crop_ratio=0.5,
            flip=True,
            prompt="sample",
            dataset_name="pascal",
    ):
        self.image_paths = natsort.natsorted(glob(os.path.join(data_dir, "*.png")) + glob(os.path.join(data_dir, "*.jpg")))
        self.mask_paths = natsort.natsorted(glob(os.path.join(data_dir, "*.npy")))
        self.depth_paths = natsort.natsorted(glob(os.path.join(data_dir, "depth", "*.npy")))
        self.dataset_name = dataset_name
        # self.use_depth = len(self.depth_paths) > 0
        self.use_depth = False
        self.train = train
        self.num_frames = num_frames
        self.mask_size = mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio
        self.prompt = prompt
        self.prompt_ids = None
        
        transforms_list = [
            A.Resize(512, 512),
            A.GaussianBlur(blur_limit=(1, 5)),
        ]
        if "face" not in dataset_name:
            max_holes, max_height, max_width, p = 8, 128, 128, 0.3
        else:
            max_holes, max_height, max_width, p = 4, 64, 64, 0.3
        transforms_list.append(A.CoarseDropout(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=0,
            mask_fill_value=0,
            p=p
        ))

        if flip:
            transforms_list.append(A.HorizontalFlip())

        self.train_transform_1 = A.Compose(
            transforms_list, 
            additional_targets={'depth': 'mask'}
        )

        rotation_range = (-10, 10) if "face" in dataset_name else (-30, 30)
        self.train_transform_2 = A.Compose(
            [
                A.Resize(512, 512),
                A.Rotate(
                    rotation_range,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                ToTensorV2(),
            ], additional_targets={'depth': 'mask'}
        )
        self.current_part_idx = 0
        self.test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()], additional_targets={'depth': 'mask'})

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = np.array(image)
        address = self.image_paths[idx]
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if self.train:
            mask = np.load(self.mask_paths[idx])
            if 'xmem' in self.dataset_name:
                mask = resize_mask_to_image(image, mask)
                if random.randint(0, 2) != 0:
                    expand_ratio = 2 if random.randint(0, 2) == 0 else 1
                    image, mask = crop_image_by_mask(image, mask, expand_ratio=expand_ratio)

            if self.use_depth:
                depth = np.load(self.depth_paths[idx])
                # depth = 1. - depth
                if "face" not in self.dataset_name:
                    if random.random() < 0.1 :
                        min_scale, max_scale = 0.5, 0.8
                        image, mask, depth = zoom_out(image, mask, depth=depth, min_scale=min_scale, max_scale=max_scale)
                result = self.train_transform_1(image=image, mask=mask, depth=depth)
                image, mask, depth = result["image"], result["mask"], result["depth"]
            else:
                if "face" not in self.dataset_name:
                    if random.random() < 0.1:
                        min_scale, max_scale = 0.5, 0.8
                        image, mask, _ = zoom_out(image, mask, min_scale=min_scale, max_scale=max_scale)
                result = self.train_transform_1(image=image, mask=mask)
                image, mask = result["image"], result["mask"]

            original_mask_size = np.where(mask == self.current_part_idx, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                x_start, x_end, y_start, y_end = get_random_crop_coordinates(
                    (self.min_crop_ratio, 1), 512, 512
                )
                aux_mask = mask[y_start:y_end, x_start:x_end]
                if self.use_depth:
                    aux_depth = depth[y_start:y_end, x_start:x_end]
                if (
                        original_mask_size == 0
                        or np.where(aux_mask == self.current_part_idx, 1, 0).sum()
                        / original_mask_size
                        > 0.3
                ):
                    mask_is_included = True

            image = image[y_start:y_end, x_start:x_end]
            if not self.use_depth:
                result = self.train_transform_2(image=image, mask=aux_mask)
                mask, image = result["mask"], result["image"]
            else:
                result = self.train_transform_2(image=image, mask=aux_mask, depth=aux_depth)
                mask, image, depth = result["mask"], result["image"], result["depth"]

            mask = torch.nn.functional.interpolate(
                mask[None, None, ...].float(),
                size=(self.mask_size, self.mask_size),
                mode="nearest",
            )[0, 0]

            self.current_part_idx = (self.current_part_idx + 1) % self.num_parts
            image = image.repeat(self.num_frames, 1, 1, 1)
            mask = mask.repeat(self.num_frames, 1, 1)
            outputs = {"image": image / 255, "mask": mask, "address": address, "prompt": self.prompt}
            if self.use_depth:
                outputs["depth"] = depth
            return outputs
        else:
            outputs = {}
            if len(self.mask_paths) > 0:
                mask = np.load(self.mask_paths[idx])
                if 'xmem' in self.dataset_name:
                    mask = resize_mask_to_image(image, mask)
                    expand_ratio = 1 if random.randint(0, 2) == 0 else 1
                    image, mask = crop_image_by_mask(image, mask, expand_ratio=expand_ratio)
                if self.use_depth:
                    depth = np.load(self.depth_paths[idx])
                    # depth = 1. - depth
                    result = self.test_transform(image=image, mask=mask, depth=depth)
                    image, mask, depth = result["image"], result["mask"], result["depth"]
                else:
                    result = self.test_transform(image=image, mask=mask)
                    image, mask = result["image"], result["mask"]
                mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].float(),
                    size=(self.mask_size, self.mask_size),
                    mode="nearest",
                )[0, 0]
                mask = mask.repeat(self.num_frames, 1, 1)
            image = image.repeat(self.num_frames, 1, 1, 1)
            outputs = {"image": image / 255, "mask": mask, "address": address, "prompt": self.prompt}
            if self.use_depth:
                outputs["depth"] = depth
            return outputs

    def __len__(self):
        return len(self.image_paths)
