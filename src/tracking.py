import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F


def initialize_tracker(pipeline):
    # pipeline.tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to('cuda')
    pipeline.tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to('cuda')

def preprocess_frames(frames, interpolate_size):
    if interpolate_size is not None:
        assert isinstance(interpolate_size, int), "interpolate_size must be an integer"
        images = torch.zeros(frames.shape[0], frames.shape[1], frames.shape[2], interpolate_size, interpolate_size).to(frames.device)
    else:
        images = torch.zeros(frames.shape[0], frames.shape[1], frames.shape[2], 512, 512).to(frames.device)
    
    for b in range(frames.shape[0]):
        for v in range(frames.shape[1]):
            x = frames[b, v].to(torch.float32)
            x = (x + 1) / 2
            if interpolate_size is not None:
                x = F.interpolate(x[None, ...], (interpolate_size, interpolate_size), mode='bilinear')[0]
            images[b, v] = x

    return images

def create_queries(batch_size, num_frames, height, width, query_frame=0, scale=None):
    ones = torch.ones(batch_size, num_frames, height, width).to('cuda')
    queries = torch.nonzero(ones[0, 0][None]==1).float().unsqueeze(0)
    queries = queries[:, :, [0, 2, 1]]
    queries = (queries+0.5) * scale if scale else queries
    queries[:, :, 0] = 1. * query_frame
    return queries

def process_queries(pipeline, images, queries, batching_ratio):
    if batching_ratio:
        pred_tracks, pred_visibility = [], []
        for i in range(batching_ratio):
            start = queries.shape[1] * i // batching_ratio
            end = queries.shape[1] * (i + 1) // batching_ratio
            queries_batch = queries[:, start:end]
            pred_track_batch, pred_visibility_batch = pipeline.tracker(images, queries_batch, backward_tracking=True)
            pred_tracks.append(pred_track_batch)
            pred_visibility.append(pred_visibility_batch)
        pred_tracks = torch.cat(pred_tracks, dim=2)
        pred_visibility = torch.cat(pred_visibility, dim=2)
    else:
        pred_tracks, pred_visibility = pipeline.tracker(images, queries, backward_tracking=True)

    return pred_tracks, pred_visibility

def postprocess_predictions(pred_tracks, pred_visibility, interpolate_size):
    pred_tracks_indices = np.round(pred_tracks[0].cpu().numpy()).astype(int)
    pred_visibility = pred_visibility[0].cpu().numpy()

    max_index = interpolate_size - 1 if interpolate_size else 511
    pred_visibility[np.max(pred_tracks_indices, axis=2) > max_index] = 0
    pred_visibility[np.min(pred_tracks_indices, axis=2) < 0] = 0

    return pred_tracks_indices, pred_visibility

def apply_tracker(pipeline, frames, query_frame=0, interpolate_size=None, batching_ratio=1):
    frames = rearrange(frames, 'b c f h w -> b f c h w')
    images = preprocess_frames(frames, interpolate_size)
    images_512 = preprocess_frames(frames, 512)

    track_on_higher_res_scale = 512 // interpolate_size if interpolate_size else None
    queries = create_queries(images.shape[0], images.shape[1], images.shape[3], images.shape[4], query_frame, scale=track_on_higher_res_scale)

    images = images_512 if track_on_higher_res_scale else images
    pred_tracks, pred_visibility = process_queries(pipeline, images, queries, batching_ratio)
    pred_tracks = pred_tracks / track_on_higher_res_scale - 0.5 if track_on_higher_res_scale else pred_tracks

    pred_tracks_indices, pred_visibility = postprocess_predictions(pred_tracks, pred_visibility, interpolate_size)
    
    return pred_tracks_indices, pred_visibility

def apply_tracker_on_all_frames(pipeline, frames):
    pred_tracks_indices, pred_visibilities = [], []
    for i in tqdm(range(frames.shape[2]), desc="Applying tracker on frames"):
        track_indices, visibility = pipeline.apply_tracker(frames, query_frame=i)
        pred_tracks_indices.append(track_indices)
        pred_visibilities.append(visibility)
    return pred_tracks_indices, pred_visibilities

def determine_window(index, length, window_size=7):
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    
    half_window = window_size // 2
    
    if index >= half_window and index < length - half_window:
        return index - half_window, index + half_window + 1
    elif index < half_window:
        return 0, window_size
    else:
        return length - window_size, length

def gather_features(was, pred_track_indice, pred_visibility, i, k, start, end):
    feats = []
    for j in range(start, end):
        if pred_visibility[j, k] > 0:
            coords = pred_track_indice[j, k]
            feats.append(was[j, :, coords[1], coords[0]])
    if feats:
        feats = torch.stack(feats, dim=0)
    else:
        feats = None
    return feats

def calculate_new_was_value(feats):
    labels = feats.argmax(axis=1)
    unique_elements, counts = np.unique(np.array(labels.cpu()), return_counts=True)
    max_label = unique_elements[np.argmax(counts)]
    # if max_label == 0:
    #     continue
    new_was_value = sum(feats[ind] for ind, label in enumerate(labels) if label == max_label)
    return new_was_value / (labels == max_label).sum()

def apply_voting(was, pred_track_indices, pred_visibilities):
    new_was = was.clone()
    for i in range(was.shape[0]):
        pred_track_indice, pred_track_visibility = pred_track_indices[i], pred_visibilities[i]
        start, end = determine_window(i, was.shape[0], 15)
        # start, end = determine_window(i, was.shape[0], 7)

        for k in range(pred_track_indice.shape[1]):
            feats = gather_features(was, pred_track_indice, pred_track_visibility, i, k, start, end)
            if feats is not None:
                new_was_value = calculate_new_was_value(feats)
                coords = pred_track_indice[i, k]
                new_was[i, :, coords[1], coords[0]] = new_was_value 
    return new_was
