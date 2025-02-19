def create_hook(pipeline, attention_layers_to_use):
    def create_nested_hook_for_attention_modules(n):
        def hook(module, input, output):
            pipeline.attention_maps[n] = [output[1], output[2]]
        return hook

    pipeline.handles = []
    for module in attention_layers_to_use:
        pipeline.handles.append(
            eval("pipeline.unet." + module).register_forward_hook(
                create_nested_hook_for_attention_modules(module)
            )
        )


def remove_hook(pipeline):
    pipeline.handles[-1].remove()
    pipeline.handles.pop()
    pipeline.attention_maps = {}


def add_perturb_hook(pipeline, modification):
    def create_nested_hook_for_perturb(modification):
        def hook(module, input, output):
            x = output[0] + modification[:, :, None]
            output = (x, *output[1:])
            return output
        return hook

    # perturb_layers = ['up_blocks[3].attentions[2].transformer_blocks[0].attn2']
    perturb_layers = [
        'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
        # 'down_blocks[0].attentions[0].transformer_blocks[0].attn2'
    ]
    # perturb_layers = ['down_blocks[0].attentions[0].transformer_blocks[0].attn2']
    for layer in perturb_layers:
        pipeline.handles.append(eval("pipeline.unet." + layer).register_forward_hook(
            create_nested_hook_for_perturb(modification)
        ))
