import matplotlib.pyplot as plt
import math

import os
import cv2
import json
import numpy as np
from skimage.morphology import disk


def calculate_iou(prediction, mask):
    intersection = prediction * mask
    union = prediction + mask - intersection
    return 100. * intersection.sum() / (union.sum() + 1e-7)


__all__ = ['batched_jaccard', 'batched_f_measure']


def batched_jaccard(y_true, y_pred, average_over_objects=True, nb_objects=None):
    """ Batch jaccard similarity for multiple instance segmentation.

    Jaccard similarity over two subsets of binary elements $A$ and $B$:

    $$
    \mathcal{J} = \\frac{A \\cap B}{A \\cup B}
    $$

    # Arguments
        y_true: Numpy Array. Array of shape (B x H x W) and type integer giving the
            ground truth of the object instance segmentation.
        y_pred: Numpy Array. Array of shape (B x H x W) and type integer giving the
            prediction of the object segmentation.
        average_over_objects: Boolean. Weather or not to average the jaccard over
            all the objects in the sequence. Default True.
        nb_objects: Integer. Number of objects in the ground truth mask. If
            `None` the value will be infered from `y_true`. Setting this value
            will speed up the computation.

    # Returns
        ndarray: Returns an array of shape (B) with the average jaccard for
            all instances at each frame if `average_over_objects=True`. If
            `average_over_objects=False` returns an array of shape (B x nObj)
            with nObj being the number of objects on `y_true`.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))

    if nb_objects is None:
        objects_ids = np.unique(y_true[(y_true < 255) & (y_true > 0)])
        nb_objects = len(objects_ids)
    else:
        objects_ids = [i + 1 for i in range(nb_objects)]
        objects_ids = np.asarray(objects_ids, dtype=np.int32)
    if nb_objects == 0:
        raise ValueError('Number of objects in y_true should be higher than 0.')
    nb_frames = len(y_true)

    jaccard = np.empty((nb_frames, nb_objects), dtype=np.float64)

    for i, obj_id in enumerate(objects_ids):
        mask_true, mask_pred = y_true == obj_id, y_pred == obj_id

        union = (mask_true | mask_pred).sum(axis=(1, 2))
        intersection = (mask_true & mask_pred).sum(axis=(1, 2))

        for j in range(nb_frames):
            if np.isclose(union[j], 0):
                jaccard[j, i] = 1.
            else:
                jaccard[j, i] = intersection[j] / union[j]

    if average_over_objects:
        jaccard = jaccard.mean(axis=1)
    return jaccard


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries. The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    # Arguments
        seg: Segments labeled from 1..k.
        width:	Width of desired bmap  <= seg.shape[1]
        height:	Height of desired bmap <= seg.shape[0]

    # Returns
        bmap (ndarray):	Binary boundary map.

    David Martin <dmartin@eecs.berkeley.edu>
    January 2003
    """

    seg = seg.astype(np.bool_)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width,
                                                                   height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def f_measure(true_mask, pred_mask, bound_th=0.008):
    """F-measure for two 2D masks.

    # Arguments
        true_mask: Numpy Array, Binary array of shape (H x W) representing the
            ground truth mask.
        pred_mask: Numpy Array. Binary array of shape (H x W) representing the
            predicted mask.
        bound_th: Float. Optional parameter to compute the F-measure. Default is
            0.008.

    # Returns
        float: F-measure.
    """
    true_mask = np.asarray(true_mask, dtype=np.bool_)
    pred_mask = np.asarray(pred_mask, dtype=np.bool_)

    assert true_mask.shape == pred_mask.shape

    bound_pix = bound_th if bound_th >= 1 else (np.ceil(
        bound_th * np.linalg.norm(true_mask.shape)))

    fg_boundary = _seg2bmap(pred_mask)
    gt_boundary = _seg2bmap(true_mask)

    fg_dil = cv2.dilate(
        fg_boundary.astype(np.uint8),
        disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(
        gt_boundary.astype(np.uint8),
        disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def batched_f_measure(y_true,
                      y_pred,
                      average_over_objects=True,
                      nb_objects=None,
                      bound_th=0.008):
    """ Batch F-measure for multiple instance segmentation.

    # Arguments
        y_true: Numpy Array. Array of shape (B x H x W) and type integer giving
            the ground truth of the object instance segmentation.
        y_pred: Numpy Array. Array of shape (B x H x W) and type integer giving
            the
            prediction of the object segmentation.
        average_over_objects: Boolean. Weather or not to average the F-measure
            over all the objects in the sequence. Default True.
        nb_objects: Integer. Number of objects in the ground truth mask. If
            `None` the value will be infered from `y_true`. Setting this value
            will speed up the computation.

    # Returns
        ndarray: Returns an array of shape (B) with the average F-measure for
            all instances at each frame if `average_over_objects=True`. If
            `average_over_objects=False` returns an array of shape (B x nObj)
            with nObj being the number of objects on `y_true`.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))

    if nb_objects is None:
        objects_ids = np.unique(y_true[(y_true < 255) & (y_true > 0)])
        nb_objects = len(objects_ids)
    else:
        objects_ids = [i + 1 for i in range(nb_objects)]
        objects_ids = np.asarray(objects_ids, dtype=np.int32)
    if nb_objects == 0:
        raise ValueError('Number of objects in y_true should be higher than 0.')
    nb_frames = len(y_true)

    f_measure_result = np.empty((nb_frames, nb_objects), dtype=np.float64)

    for i, obj_id in enumerate(objects_ids):
        for frame_id in range(nb_frames):
            gt_mask = y_true[frame_id, :, :] == obj_id
            pred_mask = y_pred[frame_id, :, :] == obj_id
            f_measure_result[frame_id, i] = f_measure(
                gt_mask, pred_mask, bound_th=bound_th)

    if average_over_objects:
        f_measure_result = f_measure_result.mean(axis=1)
    return f_measure_result


def calculate_metrics(pipeline, y_true, y_pred, gt_inds=None, frame_inds=None, output_path=None):
    # Convert y_pred to numpy array
    y_pred = y_pred.detach().cpu().numpy()
    metrics = {}

    # Filter by shared frames if gt_inds and frame_inds are provided
    if gt_inds is not None and frame_inds is not None:
        shared_frames = set(frame_inds).intersection(gt_inds)
        shared_frames = sorted(shared_frames)  # Sort shared frames (optional)
        
        pred_frame_to_index = {frame: idx for idx, frame in enumerate(frame_inds)}
        gt_frame_to_index = {frame: idx for idx, frame in enumerate(gt_inds)}
        
        y_true = y_true[[gt_frame_to_index[frame] for frame in shared_frames]]
        y_pred = y_pred[[pred_frame_to_index[frame] for frame in shared_frames]]
        
        metrics['shared_frames'] = list(shared_frames)

    # Calculate f_measure and jaccard
    f_measure = batched_f_measure(y_true, y_pred)
    jaccard = batched_jaccard(y_true, y_pred)
    
    metrics['f_measure'] = f_measure.mean()
    metrics['jaccard'] = jaccard.mean()
    
    print(f"f_measure: {f_measure.mean()}")
    print(f"jaccard: {jaccard.mean()}")

    # Calculate IoU for each part
    ious = []
    part_iou = {}
    for ind, part_name in enumerate(pipeline.part_names):
        iou_all_frames = []
        for f in range(y_true.shape[0]):
            gt = np.where(y_true[f] == ind, 1, 0).astype(np.int32)
            pred = np.where(y_pred[f] == ind, 1, 0).astype(np.int32)
            if np.all(gt == 0):
                continue
            iou_ = calculate_iou(pred, gt)
            iou_all_frames.append(iou_)
        iou = np.mean(iou_all_frames) if len(iou_all_frames) > 0 else 0
        ious.append(iou)
        
        part_iou[f"iou_{part_name}"] = iou
        # print(f"iou_{part_name}: {iou}")

    metrics['iou_per_part'] = part_iou
    metrics['mean_iou'] = np.mean(ious)

    print(f"mean iou: {metrics['mean_iou']}")

    mae = np.mean(np.abs(y_true - y_pred))
    metrics['mae'] = mae

    print(f"MAE: {mae}")

    if output_path is not None:
        output_file = os.path.join(output_path, 'metrics.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {output_file}")
        