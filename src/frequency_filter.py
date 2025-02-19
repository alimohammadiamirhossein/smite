import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
import torch_dct


def apply_filter(image, filter_type, cutoff):
    # Perform the Fourier Transform
    f_transform = torch.fft.fft2(image)
    f_transform_shifted = torch.fft.fftshift(f_transform)
    # Get the image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    # Create a mask with the same dimensions as the image
    mask = torch.ones((rows, cols), dtype=torch.complex64, device=image.device)
    if filter_type == 'low_pass':
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    elif filter_type == 'high_pass':
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
    # Apply the mask to the Fourier Transformed image
    f_transform_shifted = f_transform_shifted * mask
    # Perform the inverse Fourier Transform
    f_ishifted = torch.fft.ifftshift(f_transform_shifted)
    img_back = torch.fft.ifft2(f_ishifted)
    img_back = torch.abs(img_back)
    return img_back, f_transform_shifted

def extract_amplitude_phase(f_transform):
    amplitude = torch.abs(f_transform)
    phase = torch.angle(f_transform)
    return amplitude, phase

def frequency_loss(feat_pred, feat_gt):
    final_loss = 0
    for i in range(feat_gt.shape[0]):
        low_pass_feat_pred, low_pass_f_transform_pred = apply_filter(feat_pred[i], 'low_pass', cutoff=30)
        low_pass_feat_gt, low_pass_f_transform_gt = apply_filter(feat_gt[i], 'low_pass', cutoff=30)

        high_pass_feat_pred, high_pass_f_transform_pred = apply_filter(feat_pred[i], 'high_pass', cutoff=30)
        high_pass_feat_gt, high_pass_f_transform_gt = apply_filter(feat_gt[i], 'high_pass', cutoff=30)

        # Extract amplitude and phase
        # amplitude_orig_pred, phase_orig_pred = extract_amplitude_phase(torch.fft.fftshift(torch.fft.fft2(feat_pred[i])))
        # amplitude_orig_gt, phase_orig_gt = extract_amplitude_phase(torch.fft.fftshift(torch.fft.fft2(feat_gt[i])))

        amplitude_low_pred, phase_low_pred = extract_amplitude_phase(low_pass_f_transform_pred)
        amplitude_low_gt, phase_low_gt = extract_amplitude_phase(low_pass_f_transform_gt)

        amplitude_high_pred, phase_high_pred = extract_amplitude_phase(high_pass_f_transform_pred)
        amplitude_high_gt, phase_high_gt = extract_amplitude_phase(high_pass_f_transform_gt)

        low_loss_amplitude = torch.nn.functional.mse_loss(amplitude_low_pred, amplitude_low_gt)
        low_loss_phase = torch.nn.functional.mse_loss(phase_low_pred, phase_low_gt)
        high_loss_amplitude = torch.nn.functional.mse_loss(amplitude_high_pred, amplitude_high_gt)
        high_loss_phase = torch.nn.functional.mse_loss(phase_high_pred, phase_high_gt)

        # sum_loss = low_loss_amplitude + low_loss_phase + high_loss_amplitude + high_loss_phase
        sum_loss = low_loss_amplitude + low_loss_phase + 10*(high_loss_amplitude + high_loss_phase)
        # sum_loss = low_loss_amplitude + low_loss_phase
        final_loss += sum_loss

    return final_loss


def dct_low_pass_filter(dct_coefficients, percentage=0.3): # 2d [b c f h w]
    """
    Applies a low pass filter to the given DCT coefficients.

    :param dct_coefficients: 2D tensor of DCT coefficients
    :param percentage: percentage of coefficients to keep (between 0 and 1)
    :return: 2D tensor of DCT coefficients after applying the low pass filter
    """
    # Determine the cutoff indices for both dimensions
    cutoff_x = int(dct_coefficients.shape[-2] * percentage)
    cutoff_y = int(dct_coefficients.shape[-1] * percentage)

    # Create a mask with the same shape as the DCT coefficients
    mask = torch.zeros_like(dct_coefficients)
    # Set the top-left corner of the mask to 1 (the low-frequency area)
    mask[:, :, :, :cutoff_x, :cutoff_y] = 1

    return mask

def dct_loss(feat_pred, feat_gt, low_pass=True, threshold=0.2):
    final_loss = 0
    feat_pred = rearrange(feat_pred.unsqueeze(0), 'b f c h w -> b c f h w')
    feat_gt = rearrange(feat_gt.unsqueeze(0), 'b f c h w -> b c f h w')
    pred_freq = torch_dct.dct_3d(feat_pred, norm='ortho')
    gt_freq = torch_dct.dct_3d(feat_gt, norm='ortho')
    
    mask = torch.zeros_like(pred_freq)
    mask = dct_low_pass_filter(mask, percentage=threshold)
    # mask = dct_low_pass_filter(mask, percentage=0.8)
    # mask = dct_low_pass_filter(mask, percentage=0.99999)

    if low_pass:
        low_pass_pred_freq = pred_freq * mask
        low_pass_gt_freq = gt_freq * mask
        # low_freq_loss = F.mse_loss(low_pass_pred_freq, low_pass_gt_freq)
        # L1 loss
        low_freq_loss = F.l1_loss(low_pass_pred_freq, low_pass_gt_freq)
        final_loss += low_freq_loss
        low_pass_pred = torch_dct.idct_3d(low_pass_pred_freq, norm='ortho')
        # low_pass_pred = torch_dct.idct_3d(pred_freq, norm='ortho')
        transformed_feat = rearrange(low_pass_pred, 'b c f h w -> b f c h w')[0]
    
        # low_loss = F.mse_loss(low_pass_freq, feat_gt)
    else:
        high_pass_pred_freq = pred_freq * (1 - mask)
        high_pass_gt_freq = gt_freq * (1 - mask)
        high_freq_loss = F.mse_loss(high_pass_pred_freq, high_pass_gt_freq)
        final_loss += high_freq_loss
        high_pass_pred = torch_dct.idct_3d(high_pass_pred_freq, norm='ortho')
        transformed_feat = rearrange(high_pass_pred, 'b c f h w -> b f c h w')[0]
    
    return final_loss, transformed_feat