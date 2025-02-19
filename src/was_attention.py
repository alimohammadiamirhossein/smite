import torch
import math
from einops import rearrange
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from .data_processing import batch_to_head_dim, head_to_batch_dim

def get_attention_map(
        pipeline,
        raw_attention_maps,
        output_size=256,
        token_ids=(2,),
        average_layers=True,
        train=True,
        apply_softmax=True,
        video_length=1,
        guidance_scale=0.0,
        **kwargs,
):
    cross_attention_maps = {}
    self_attention_maps = {}
    resnets = {}
    for f in range(video_length):
        for layer in raw_attention_maps:
            if layer.endswith("attn2"):  # cross attentions
                query, key = raw_attention_maps[layer]
                head_size = query.shape[0] // (video_length * 2) if guidance_scale > 1 else query.shape[0] // video_length
                query, key = batch_to_head_dim(query, head_size), batch_to_head_dim(key, head_size)
                query = rearrange(query, "(b l) d c -> b l d c", l=video_length)
                key = rearrange(key, "(b l) d c -> b l d c", l=video_length)
                query = query[:, f]
                key = key[:, f]
                query, key = head_to_batch_dim(query, head_size), head_to_batch_dim(key, head_size)

                attention_score = torch.baddbmm(
                    torch.empty(
                        query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                    ),
                    query, key.transpose(-1, -2), beta=0, alpha=0.125,
                )

                split_attention_maps = torch.stack(attention_score.chunk(2), dim=0) if guidance_scale > 1 \
                    else attention_score.unsqueeze(0)

                _, channel, img_embed_len, text_embed_len = split_attention_maps.shape

                dim1 = 2 if guidance_scale > 1 else 1
                if train:
                    if apply_softmax:
                        reshaped_split_attention_maps = (
                            split_attention_maps[:, :, :, torch.tensor(list(token_ids))]
                                .softmax(dim=-1)
                                .reshape(
                                dim1,  # because of chunk
                                channel,
                                int(math.sqrt(img_embed_len)),
                                int(math.sqrt(img_embed_len)),
                                len(token_ids),
                            )
                                .permute(0, 1, 4, 2, 3)
                        )
                    else:
                        reshaped_split_attention_maps = (
                            split_attention_maps[:, :, :, torch.tensor(list(token_ids))]
                                .reshape(
                                dim1,  # because of chunk
                                channel,
                                int(math.sqrt(img_embed_len)),
                                int(math.sqrt(img_embed_len)),
                                len(token_ids),
                            )
                                .permute(0, 1, 4, 2, 3)
                        )
                else:
                    reshaped_split_attention_maps = (
                        split_attention_maps.softmax(dim=-1)[
                        :, :, :, torch.tensor(list(token_ids))
                        ]
                            .reshape(
                            dim1,  # because of chunk
                            channel,
                            int(math.sqrt(img_embed_len)),
                            int(math.sqrt(img_embed_len)),
                            len(token_ids),
                        )
                            .permute(0, 1, 4, 2, 3)
                    )

                ca_scale = reshaped_split_attention_maps.shape[-1]
                if guidance_scale > 1:
                    resized_reshaped_split_attention_maps_0 = (
                        torch.nn.functional.interpolate(
                            reshaped_split_attention_maps[0],
                            size=output_size,
                            mode="bilinear",
                        ).mean(dim=0)
                    )
                    resized_reshaped_split_attention_maps_1 = (
                        torch.nn.functional.interpolate(
                            reshaped_split_attention_maps[1],
                            size=output_size,
                            mode="bilinear",
                        ).mean(dim=0)
                    )
                else:
                    resized_reshaped_split_attention_maps_0 = (
                        torch.nn.functional.interpolate(
                            reshaped_split_attention_maps[0],
                            size=output_size,
                            mode="bilinear",
                        ).mean(dim=0)
                    )
                    resized_reshaped_split_attention_maps_1 = (
                        torch.nn.functional.interpolate(
                            reshaped_split_attention_maps[0],
                            size=output_size,
                            mode="bilinear",
                        ).mean(dim=0)
                    )
                resized_reshaped_split_attention_maps = torch.stack(
                    [
                        resized_reshaped_split_attention_maps_0,
                        resized_reshaped_split_attention_maps_1,
                    ],
                    dim=0,
                )
                resized_reshaped_split_attention_maps = resized_reshaped_split_attention_maps #* (ca_scale/64.)
                cross_attention_maps[f'layer_{f}'] = cross_attention_maps.get(f'layer_{f}', {})
                cross_attention_maps[f'layer_{f}'][layer] = resized_reshaped_split_attention_maps

            elif layer.endswith("attn1"):  # self attentions
                query, key = raw_attention_maps[layer]

                inflated_unet = kwargs.get("inflated_unet", True)
                
                if inflated_unet:
                    head_size = query.shape[0] // 2 if guidance_scale > 1 else query.shape[0]
                else:
                    head_size = query.shape[0] // (video_length*2) if guidance_scale > 1 else query.shape[0] // video_length
                query, key = batch_to_head_dim(query, head_size), batch_to_head_dim(key, head_size)
        
                if inflated_unet:
                    query = rearrange(query, "b (l d) c -> b l d c", l=video_length)
                    query = query[:, f]
                    key = rearrange(key, "b (l d) c -> b l d c", l=video_length)
                    key = key[:, f]
                else:
                    query = rearrange(query, "(b l) d c -> b l d c", l=video_length)
                    query = query[:, f]
                    key = rearrange(key, "(b l) d c -> b l d c", l=video_length)
                    key = key[:, f]

                ## uncomment this when you set self attention's inter_frame to True ##
                # key = rearrange(key, "b (l d) c -> b l d c", l=2)
                # key = key[:, -1]
                ############################################
                query, key = head_to_batch_dim(query, head_size), head_to_batch_dim(key, head_size)

                attention_score = torch.baddbmm(
                    torch.empty(
                        query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                    ),
                    query, key.transpose(-1, -2), beta=0, alpha=0.125,
                )

                channel, img_embed_len, img_embed_len = attention_score.shape

                split_attention_maps = attention_score[channel // 2 :] if guidance_scale > 1 else attention_score
                ## changed this line ## to check self attention on the unconditional part
                # split_attention_maps = attention_score[: channel // 2]

                dim1 = channel // 2 if guidance_scale > 1 else channel
                reshaped_split_attention_maps = (
                    split_attention_maps.softmax(dim=-1)
                        .reshape(
                        # 2,  # because of chunk
                        dim1,
                        int(math.sqrt(img_embed_len)),
                        int(math.sqrt(img_embed_len)),
                        img_embed_len,
                    )
                        .permute(0, 3, 1, 2)
                )
                resized_reshaped_split_attention_maps = torch.nn.functional.interpolate(
                    reshaped_split_attention_maps, size=output_size, mode="bilinear"
                )
                resized_reshaped_split_attention_maps = (
                    resized_reshaped_split_attention_maps.mean(dim=0)
                )
                # self_attention_maps[layer] = resized_reshaped_split_attention_maps
                self_attention_maps[f'layer_{f}'] = self_attention_maps.get(f'layer_{f}', {})
                self_attention_maps[f'layer_{f}'][layer] = resized_reshaped_split_attention_maps

            elif layer.endswith("conv1") or layer.endswith("conv2"):  # resnets
                resnets[layer] = raw_attention_maps[layer].detach()
    sd_cross_attention_maps_total = []
    sd_cross_attention_maps = [None, None]
    sd_self_attention_maps_total = []

    for f in cross_attention_maps.keys():
        cross_attention_map = cross_attention_maps[f]
        if len(cross_attention_maps.values()) > 0:
            if average_layers:
                sd_cross_attention_map = (
                    torch.stack(list(cross_attention_map.values()), dim=0)
                        .mean(dim=0)
                        .to(pipeline.device)
                )
            else:
                sd_cross_attention_map = torch.stack(
                    list(cross_attention_map.values()), dim=1
                ).to(pipeline.device)
            sd_cross_attention_maps_total.append(sd_cross_attention_map)

    sd_cross_attention_maps_total = torch.stack(sd_cross_attention_maps_total, dim=0)
    sd_cross_attention_maps_total = sd_cross_attention_maps_total.permute(1, 0 ,2, 3, 4)
    sd_cross_attention_maps_total = sd_cross_attention_maps_total[0] if (sd_cross_attention_maps_total.shape[0] == 1) \
        else sd_cross_attention_maps_total

    for f in self_attention_maps.keys():
        self_attention_map = self_attention_maps[f]
        if len(self_attention_map.values()) > 0:
            self_attention_map = torch.stack(
                list(self_attention_map.values()), dim=0
            ).mean(dim=0)
            sd_self_attention_maps_total.append(self_attention_map)

    sd_self_attention_maps_total = torch.stack(sd_self_attention_maps_total, dim=0)

    if len(resnets.values()) > 0:
        r = list(map(lambda x: x.to(pipeline.device), list(resnets.values())))

    del attention_score, query, key, split_attention_maps, reshaped_split_attention_maps, resized_reshaped_split_attention_maps

    return (
        sd_cross_attention_maps_total[0],
        sd_cross_attention_maps_total[1],
        sd_self_attention_maps_total,
    )


def calculate_was_mask(pipeline,
                                    sd_cross_attention_maps2_all_frames,
                                    sd_self_attention_maps,
                                    mask_patch_size=64,
                                    use_cross_attention_only=False
                                    ):
        final_attention_map = torch.zeros(sd_cross_attention_maps2_all_frames.shape[0],
                                          sd_cross_attention_maps2_all_frames.shape[1],
                                          mask_patch_size, mask_patch_size).to(sd_cross_attention_maps2_all_frames.device)

        thetas = [1] * pipeline.num_parts

        for f in range(sd_cross_attention_maps2_all_frames.shape[0]):
            sd_cross_attention_maps2 = sd_cross_attention_maps2_all_frames[f]
            # sd_cross_attention_maps2 = sd_cross_attention_maps2_all_frames[0]
            # sd_cross_attention_maps2 = sd_cross_attention_maps2_all_frames.mean(dim=0)
            sd_self_attention_map = sd_self_attention_maps[f]
            sd_cross_flat = sd_cross_attention_maps2.flatten(1, 2)
            max_values = sd_cross_flat.max(dim=1).values
            min_values = sd_cross_flat.min(dim=1).values

            new_sd_cross_flat = torch.zeros_like(sd_cross_flat)
            new_sd_cross_flat[1:] = sd_cross_flat[1:]
            new_sd_cross_flat[0] = torch.where(
                sd_cross_flat[0]
                > sd_cross_flat[0].mean(),
                sd_cross_flat[0],
                0,
                )
            sd_cross_flat = new_sd_cross_flat

            for idx, mask_id in enumerate(range(pipeline.num_parts)):
                if use_cross_attention_only:
                    avg_self_attention_map = sd_cross_attention_maps2[idx]
                else:
                    avg_self_attention_map = (
                            sd_cross_flat[idx][..., None, None]
                            * (sd_self_attention_map ** thetas[idx])
                    ).sum(dim=0)

                avg_self_attention_map = F.interpolate(
                    avg_self_attention_map[None, None, ...],
                    mask_patch_size,
                    mode="bilinear",
                )[0, 0]
                avg_self_attention_map_min = avg_self_attention_map.min()
                avg_self_attention_map_max = avg_self_attention_map.max()
                coef = (
                               avg_self_attention_map_max - avg_self_attention_map_min
                       ) / (max_values[mask_id] - min_values[mask_id])

                final_attention_map[f][idx] = (avg_self_attention_map / coef) + (
                        min_values[mask_id] - avg_self_attention_map_min / coef
                )

        return final_attention_map


def calculate_multi_was_mask(pipeline,
                                  maps,
                                  crop_coords,
                                  output_size=64,
                                  use_cross_attention_only=False
                                  ):
    random_crop = crop_coords[0]
    sd_cross_attention_maps2_all_frames = maps[f'{random_crop}']['cross_attention_maps']
    final_attention_map = torch.zeros(sd_cross_attention_maps2_all_frames.shape[0],
                                      sd_cross_attention_maps2_all_frames.shape[1],
                                      output_size, output_size).to(sd_cross_attention_maps2_all_frames.device)
    aux_attention_map = torch.zeros_like(final_attention_map).to(sd_cross_attention_maps2_all_frames.device)

    thetas = [1] * pipeline.num_parts
    ratio = 512 // output_size
    mask_patch_size = crop_coords[0][1] - crop_coords[0][0]

    for crop_coord in crop_coords:
        y_start, y_end, x_start, x_end = crop_coord
        mask_y_start, mask_y_end, mask_x_start, mask_x_end = (y_start // ratio, y_end // ratio,
                                                              x_start // ratio, x_end // ratio)
                                                              
        sd_cross_attention_maps2_all_frames = maps[f'{crop_coord}']['cross_attention_maps']
        sd_self_attention_maps = maps[f'{crop_coord}']['self_attention_maps']
        
        for f in range(sd_cross_attention_maps2_all_frames.shape[0]):
            sd_cross_attention_maps2 = sd_cross_attention_maps2_all_frames[f]
            sd_self_attention_map = sd_self_attention_maps[f]
            sd_cross_flat = sd_cross_attention_maps2.flatten(1, 2)
            max_values = sd_cross_flat.max(dim=1).values
            min_values = sd_cross_flat.min(dim=1).values

            new_sd_cross_flat = torch.zeros_like(sd_cross_flat)
            new_sd_cross_flat[1:] = sd_cross_flat[1:]
            new_sd_cross_flat[0] = torch.where(
                sd_cross_flat[0]
                > sd_cross_flat[0].mean(),
                sd_cross_flat[0],
                0,
                )
            sd_cross_flat = new_sd_cross_flat

            for idx, mask_id in enumerate(range(pipeline.num_parts)):
                if use_cross_attention_only:
                    avg_self_attention_map = sd_cross_attention_maps2[idx]
                else:
                    avg_self_attention_map = (
                            sd_cross_flat[idx][..., None, None]
                            * (sd_self_attention_map ** thetas[idx])
                    ).sum(dim=0)
                avg_self_attention_map = avg_self_attention_map.to(torch.float32)
                avg_self_attention_map = F.interpolate(
                    avg_self_attention_map[None, None, ...],
                    mask_patch_size,
                    mode="bilinear",
                )[0, 0]
                avg_self_attention_map_min = avg_self_attention_map.min()
                avg_self_attention_map_max = avg_self_attention_map.max()
                coef = (
                               avg_self_attention_map_max - avg_self_attention_map_min
                       ) / (max_values[mask_id] - min_values[mask_id])

                final_attention_map[f][idx][mask_y_start:mask_y_end, mask_x_start:mask_x_end] = (avg_self_attention_map / coef) + (
                        min_values[mask_id] - avg_self_attention_map_min / coef
                )
                aux_attention_map[f][idx][mask_y_start:mask_y_end, mask_x_start:mask_x_end] += \
                    torch.ones_like(avg_self_attention_map, dtype=torch.uint8)

    final_attention_map = final_attention_map / aux_attention_map
    return final_attention_map
