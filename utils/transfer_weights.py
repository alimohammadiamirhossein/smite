import torch.nn as nn


def transfer_unets(unet3d, unet2d, is_controlnet=False):
    # for name, param in unet2d.named_parameters():
    #     print(name, param.shape)

    unet3d.conv_in.weight = nn.Parameter(unet2d.conv_in.weight.clone())
    unet3d.conv_in.bias = nn.Parameter(unet2d.conv_in.bias.clone())
    unet3d.time_embedding.linear_1.weight = nn.Parameter(unet2d.time_embedding.linear_1.weight.clone())
    unet3d.time_embedding.linear_1.bias = nn.Parameter(unet2d.time_embedding.linear_1.bias.clone())
    unet3d.time_embedding.linear_2.weight = nn.Parameter(unet2d.time_embedding.linear_2.weight.clone())
    unet3d.time_embedding.linear_2.bias = nn.Parameter(unet2d.time_embedding.linear_2.bias.clone())

    for i in range(4):
        # down blocks
        if i < 3:
            for j in range(2):
                unet3d.down_blocks[i].attentions[j].norm.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].norm.weight.clone())
                unet3d.down_blocks[i].attentions[j].norm.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].norm.bias.clone())
                unet3d.down_blocks[i].attentions[j].proj_in.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].proj_in.weight.clone())
                unet3d.down_blocks[i].attentions[j].proj_in.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].proj_in.bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm1.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm1.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm1.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm1.bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm2.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm2.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm2.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm2.bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].bias.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm3.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm3.weight.clone())
                unet3d.down_blocks[i].attentions[j].transformer_blocks[0].norm3.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].transformer_blocks[0].norm3.bias.clone())
                unet3d.down_blocks[i].attentions[j].proj_out.weight = nn.Parameter(unet2d.down_blocks[i].attentions[j].proj_out.weight.clone())
                unet3d.down_blocks[i].attentions[j].proj_out.bias = nn.Parameter(unet2d.down_blocks[i].attentions[j].proj_out.bias.clone())
            unet3d.down_blocks[i].downsamplers[0].conv.weight = nn.Parameter(unet2d.down_blocks[i].downsamplers[0].conv.weight.clone())
            unet3d.down_blocks[i].downsamplers[0].conv.bias = nn.Parameter(unet2d.down_blocks[i].downsamplers[0].conv.bias.clone())



        for j in range(2):
            unet3d.down_blocks[i].resnets[j].norm1.weight = nn.Parameter(unet2d.down_blocks[i].resnets[j].norm1.weight.clone())
            unet3d.down_blocks[i].resnets[j].norm1.bias = nn.Parameter(unet2d.down_blocks[i].resnets[j].norm1.bias.clone())
            unet3d.down_blocks[i].resnets[j].conv1.weight = nn.Parameter(unet2d.down_blocks[i].resnets[j].conv1.weight.clone())
            unet3d.down_blocks[i].resnets[j].conv1.bias = nn.Parameter(unet2d.down_blocks[i].resnets[j].conv1.bias.clone())
            unet3d.down_blocks[i].resnets[j].time_emb_proj.weight = nn.Parameter(unet2d.down_blocks[i].resnets[j].time_emb_proj.weight.clone())
            unet3d.down_blocks[i].resnets[j].time_emb_proj.bias = nn.Parameter(unet2d.down_blocks[i].resnets[j].time_emb_proj.bias.clone())
            unet3d.down_blocks[i].resnets[j].norm2.weight = nn.Parameter(unet2d.down_blocks[i].resnets[j].norm2.weight.clone())
            unet3d.down_blocks[i].resnets[j].norm2.bias = nn.Parameter(unet2d.down_blocks[i].resnets[j].norm2.bias.clone())
            unet3d.down_blocks[i].resnets[j].conv2.weight = nn.Parameter(unet2d.down_blocks[i].resnets[j].conv2.weight.clone())
            unet3d.down_blocks[i].resnets[j].conv2.bias = nn.Parameter(unet2d.down_blocks[i].resnets[j].conv2.bias.clone())

        if 3 > i > 0:
            unet3d.down_blocks[i].resnets[0].conv_shortcut.weight = nn.Parameter(unet2d.down_blocks[i].resnets[0].conv_shortcut.weight.clone())
            unet3d.down_blocks[i].resnets[0].conv_shortcut.bias = nn.Parameter(unet2d.down_blocks[i].resnets[0].conv_shortcut.bias.clone())


        # up blocks
        if i > 0 and not is_controlnet:
            for j in range(3):
                # Attention Transformer Blocks
                unet3d.up_blocks[i].attentions[j].norm.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].norm.weight.clone())
                unet3d.up_blocks[i].attentions[j].norm.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].norm.bias.clone())
                unet3d.up_blocks[i].attentions[j].proj_in.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].proj_in.weight.clone())
                unet3d.up_blocks[i].attentions[j].proj_in.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].proj_in.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_q.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_k.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_v.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn1.to_out[0].bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm1.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm1.weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm1.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm1.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_q.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_k.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.weight.clone())
                # unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_v.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].attn2.to_out[0].bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm2.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm2.weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm2.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm2.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[0].proj.bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].ff.net[2].bias.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm3.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm3.weight.clone())
                unet3d.up_blocks[i].attentions[j].transformer_blocks[0].norm3.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].transformer_blocks[0].norm3.bias.clone())
                unet3d.up_blocks[i].attentions[j].proj_out.weight = nn.Parameter(unet2d.up_blocks[i].attentions[j].proj_out.weight.clone())
                unet3d.up_blocks[i].attentions[j].proj_out.bias = nn.Parameter(unet2d.up_blocks[i].attentions[j].proj_out.bias.clone())

        if i < 3 and not is_controlnet:
            unet3d.up_blocks[i].upsamplers[0].conv.weight = nn.Parameter(unet2d.up_blocks[i].upsamplers[0].conv.weight.clone())
            unet3d.up_blocks[i].upsamplers[0].conv.bias = nn.Parameter(unet2d.up_blocks[i].upsamplers[0].conv.bias.clone())

        for j in range(3):
            if is_controlnet:
                break
            unet3d.up_blocks[i].resnets[j].norm1.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].norm1.weight.clone())
            unet3d.up_blocks[i].resnets[j].norm1.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].norm1.bias.clone())
            unet3d.up_blocks[i].resnets[j].conv1.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv1.weight.clone())
            unet3d.up_blocks[i].resnets[j].conv1.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv1.bias.clone())
            unet3d.up_blocks[i].resnets[j].time_emb_proj.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].time_emb_proj.weight.clone())
            unet3d.up_blocks[i].resnets[j].time_emb_proj.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].time_emb_proj.bias.clone())
            unet3d.up_blocks[i].resnets[j].norm2.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].norm2.weight.clone())
            unet3d.up_blocks[i].resnets[j].norm2.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].norm2.bias.clone())
            unet3d.up_blocks[i].resnets[j].conv2.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv2.weight.clone())
            unet3d.up_blocks[i].resnets[j].conv2.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv2.bias.clone())
            unet3d.up_blocks[i].resnets[j].conv_shortcut.weight = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv_shortcut.weight.clone())
            unet3d.up_blocks[i].resnets[j].conv_shortcut.bias = nn.Parameter(unet2d.up_blocks[i].resnets[j].conv_shortcut.bias.clone())

    # mid block
    unet3d.mid_block.attentions[0].norm.weight = nn.Parameter(unet2d.mid_block.attentions[0].norm.weight.clone())
    unet3d.mid_block.attentions[0].norm.bias = nn.Parameter(unet2d.mid_block.attentions[0].norm.bias.clone())
    unet3d.mid_block.attentions[0].proj_in.weight = nn.Parameter(unet2d.mid_block.attentions[0].proj_in.weight.clone())
    unet3d.mid_block.attentions[0].proj_in.bias = nn.Parameter(unet2d.mid_block.attentions[0].proj_in.bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn1.to_k.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn1.to_k.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn1.to_v.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn1.to_v.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn1.to_out[0].weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn1.to_out[0].weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn1.to_out[0].bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn1.to_out[0].bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm1.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm1.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm1.bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm1.bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn2.to_q.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn2.to_q.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn2.to_k.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn2.to_k.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn2.to_out[0].weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn2.to_out[0].weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].attn2.to_out[0].bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].attn2.to_out[0].bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm2.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm2.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm2.bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm2.bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].ff.net[0].proj.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].ff.net[0].proj.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].ff.net[0].proj.bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].ff.net[0].proj.bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].ff.net[2].weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].ff.net[2].weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].ff.net[2].bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].ff.net[2].bias.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm3.weight = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm3.weight.clone())
    unet3d.mid_block.attentions[0].transformer_blocks[0].norm3.bias = nn.Parameter(unet2d.mid_block.attentions[0].transformer_blocks[0].norm3.bias.clone())
    unet3d.mid_block.attentions[0].proj_out.weight = nn.Parameter(unet2d.mid_block.attentions[0].proj_out.weight.clone())
    unet3d.mid_block.attentions[0].proj_out.bias = nn.Parameter(unet2d.mid_block.attentions[0].proj_out.bias.clone())

    for j in range(2):
        unet3d.mid_block.resnets[j].norm1.weight = nn.Parameter(unet2d.mid_block.resnets[j].norm1.weight.clone())
        unet3d.mid_block.resnets[j].norm1.bias = nn.Parameter(unet2d.mid_block.resnets[j].norm1.bias.clone())
        unet3d.mid_block.resnets[j].conv1.weight = nn.Parameter(unet2d.mid_block.resnets[j].conv1.weight.clone())
        unet3d.mid_block.resnets[j].conv1.bias = nn.Parameter(unet2d.mid_block.resnets[j].conv1.bias.clone())
        unet3d.mid_block.resnets[j].time_emb_proj.weight = nn.Parameter(unet2d.mid_block.resnets[j].time_emb_proj.weight.clone())
        unet3d.mid_block.resnets[j].time_emb_proj.bias = nn.Parameter(unet2d.mid_block.resnets[j].time_emb_proj.bias.clone())
        unet3d.mid_block.resnets[j].norm2.weight = nn.Parameter(unet2d.mid_block.resnets[j].norm2.weight.clone())
        unet3d.mid_block.resnets[j].norm2.bias = nn.Parameter(unet2d.mid_block.resnets[j].norm2.bias.clone())
        unet3d.mid_block.resnets[j].conv2.weight = nn.Parameter(unet2d.mid_block.resnets[j].conv2.weight.clone())
        unet3d.mid_block.resnets[j].conv2.bias = nn.Parameter(unet2d.mid_block.resnets[j].conv2.bias.clone())
        unet3d.mid_block.resnets[j].conv_shortcut = unet2d.mid_block.resnets[j].conv_shortcut

    if is_controlnet:
        unet3d.controlnet_cond_embedding.conv_in.weight = nn.Parameter(unet2d.controlnet_cond_embedding.conv_in.weight.clone())
        unet3d.controlnet_cond_embedding.conv_in.bias = nn.Parameter(unet2d.controlnet_cond_embedding.conv_in.bias.clone())
        for m in range(6):
            unet3d.controlnet_cond_embedding.blocks[m].weight = nn.Parameter(unet2d.controlnet_cond_embedding.blocks[m].weight.clone())
            unet3d.controlnet_cond_embedding.blocks[m].bias = nn.Parameter(unet2d.controlnet_cond_embedding.blocks[m].bias.clone())
        unet3d.controlnet_cond_embedding.conv_out.weight = nn.Parameter(unet2d.controlnet_cond_embedding.conv_out.weight.clone())
        unet3d.controlnet_cond_embedding.conv_out.bias = nn.Parameter(unet2d.controlnet_cond_embedding.conv_out.bias.clone())
        for m in range(12):
            unet3d.controlnet_down_blocks[m].weight = nn.Parameter(unet2d.controlnet_down_blocks[m].weight.clone())
            unet3d.controlnet_down_blocks[m].bias = nn.Parameter(unet2d.controlnet_down_blocks[m].bias.clone())
        unet3d.controlnet_mid_block.weight = nn.Parameter(unet2d.controlnet_mid_block.weight.clone())
        unet3d.controlnet_mid_block.bias = nn.Parameter(unet2d.controlnet_mid_block.bias.clone())
        return unet3d

    unet3d.conv_norm_out.weight = nn.Parameter(unet2d.conv_norm_out.weight.clone())
    unet3d.conv_norm_out.bias = nn.Parameter(unet2d.conv_norm_out.bias.clone())
    unet3d.conv_out.weight = nn.Parameter(unet2d.conv_out.weight.clone())
    unet3d.conv_out.bias = nn.Parameter(unet2d.conv_out.bias.clone())

    return unet3d