import torch

def create_slices(length, slice_size):
  """Creates slices of size `slice_size` covering the entire range from 0 to `length`."""
  slices = []
  start = 0
  while start < length:
    end = min(start + slice_size, length)
    slices.append(slice(start, end))
    start = end
  return slices

def create_slices2(length, slice_size=18, overlap=2):
  """Creates slices of size `slice_size` covering the entire range from 0 to `length`."""
  slices = []
  start = 0
  while start < length:
    if start > 0:
        start -= overlap
    end = min(start + slice_size, length)
    if end == length and end - slice_size > 0:
        start = end - slice_size
    slices.append(slice(start, end))
    start = end
  return slices

def slicing_latents(slices, ind, frames, depth_input):
    saved_interval = 2
    if ind < len(slices) - 1:
        saved_interval = slices[ind].stop - slices[ind+1].start
    slice_ = slices[ind]
    sliced_video_lenght = slice_.stop - slice_.start
    sliced_frames = frames[slice_]
    sliced_depth = depth_input[slice_] if depth_input is not None else None
    return sliced_frames, sliced_depth, sliced_video_lenght, saved_interval

def process_before_slice_saving(outputs_, slices, ind, saved_interval):
    saved_was = outputs_["maps"]["was_64_attention_maps"][-saved_interval:]
    cross_attn = outputs_["maps"]["cross_attention_maps"].cpu()
    self_attn = outputs_["maps"]["self_attention_maps"].cpu()
    if ind < len(slices) - 1:
        cross_attn = cross_attn[:-saved_interval or None]
        self_attn = self_attn[:-saved_interval or None]
    return saved_was, cross_attn, self_attn

def _apply_slicing_with_crops(slices, frames, depth_input, pipeline, crop_coord, args_to_pass, **kwargs):
    saved_was = None
    map_ = {'cross_attention_maps': [], 'self_attention_maps': []}

    for ind, slice_ in enumerate(slices):
        print(f"Starting to process frames from {slice_.start} to {slice_.stop}")
        sliced_frames, sliced_depth, sliced_video_lenght, saved_interval = slicing_latents(slices, ind, frames, depth_input)
        kwargs['depths'] = sliced_depth
        
        pipeline.clean_features()
        outputs_ = pipeline._inference(video_length=sliced_video_lenght, frames=sliced_frames, saved_was=saved_was,
                        crop_coord=crop_coord, **args_to_pass, **kwargs)

        saved_was, cross_attn, self_attn = process_before_slice_saving(outputs_, slices, ind, saved_interval)

        map_['cross_attention_maps'].append(cross_attn)
        map_['self_attention_maps'].append(self_attn)

        if ind  < len(slices) - 1:
            del outputs_

    map_['cross_attention_maps'] = torch.cat(map_['cross_attention_maps'], dim=0) # concat all of slices
    map_['self_attention_maps'] = torch.cat(map_['self_attention_maps'], dim=0) # concat all of slices

    return map_

def _apply_slicing_without_crops(slices, frames, depth_input, pipeline, args_to_pass, **kwargs):
    was_full, cross_attn_full, self_attn_full = [], [], []
    saved_was = None

    for ind, slice_ in enumerate(slices):
        print(f"Starting to process frames from {slice_.start} to {slice_.stop}")
        sliced_frames, sliced_depth, sliced_video_lenght, saved_interval = slicing_latents(slices, ind, frames, depth_input)
        kwargs['depths'] = sliced_depth

        output_ = pipeline._inference(video_length=sliced_video_lenght, frames=sliced_frames, 
                                      saved_was=saved_was, **args_to_pass, **kwargs)

        saved_was = output_["maps"]["was_64_attention_maps"][-saved_interval:]
        ws_512 = output_["maps"]["was_attention_maps"].cpu()
        ws_512 = ws_512[:-saved_interval or None] if ind < len(slices) - 1 else ws_512
        cross_attn = output_["maps"]["cross_attention_maps"][:-saved_interval or None]
        self_attn = output_["maps"]["self_attention_maps"][:-saved_interval or None]
        was_full.append(ws_512)
        cross_attn_full.append(cross_attn)
        # self_attn_full.append(self_attn)

        if ind  < len(slices) - 1:
            del output_, sliced_frames

    new_maps = {"cross_attention_maps": torch.cat(cross_attn_full, dim=0),
                # "self_attention_maps": torch.cat(self_attn_full, dim=0),
                 "was_attention_maps": torch.cat(was_full, dim=0)}

    return new_maps