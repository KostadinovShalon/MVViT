import torch


def apply_multiview_layer(x, layer, *args, **kwargs):
    x = x.transpose(0, 1).contiguous()  # V B
    if isinstance(layer, list):
        xvs = []
        for xv in x:
            for l in layer:
                xv = l(xv, *args, **kwargs)
            xvs.append(xv)
        x = torch.stack(xvs)
    else:
        x = torch.stack([layer(xv, *args, **kwargs) for xv in x])
    x = x.transpose(0, 1).contiguous()  # B V
    return x


def apply_multiview_patch_embed_layer(x, patch_embed_layer):
    x = x.transpose(0, 1).contiguous()  # V B
    xs = []
    hw_shape = None
    for xv in x:
        _x, hw_shape = patch_embed_layer(xv)
        xs.append(_x)
    x = torch.stack(xs)
    x = x.transpose(0, 1).contiguous()  # B V
    return x, hw_shape


def apply_multiview_swin_sequence_layer(x, hw_shape, stage):
    x = x.transpose(0, 1).contiguous()  # V B
    xs = []
    _hw_shape = None  # v-list of 2-tuples
    outs = []
    out_hw_shape = None
    for xv in x:
        _x, _hw_shape, _out, out_hw_shape = stage(xv, hw_shape)
        xs.append(_x)
        outs.append(_out)
    x = torch.stack(xs)
    out = torch.stack(outs)
    x = x.transpose(0, 1).contiguous()  # B V
    out = out.transpose(0, 1).contiguous()  # B V
    return x, _hw_shape, out, out_hw_shape
#
# def apply_multiview_layer(x, layer):
#     b, v, c, h, w = x.shape
#     x = x.flatten(0, 1)  # B.V
#     if isinstance(layer, list):
#         # xvs = []
#         # for xv in x:
#         for l in layer:
#             x = l(x)
#         #     xvs.append(xv)
#         # x = torch.stack(xvs)
#     else:
#         x = layer(x)
#         # x = torch.stack([layer(xv) for xv in x])
#     # x = x.transpose(0, 1).contiguous()  # B V
#     x = x.view(b, v, *x.shape[-3:])
#     return x