import torch


def apply_multiview_layer(x, layer):
    x = x.transpose(0, 1).contiguous()  # V B
    if isinstance(layer, list):
        xvs = []
        for xv in x:
            for l in layer:
                xv = l(xv)
            xvs.append(xv)
        x = torch.stack(xvs)
    else:
        x = torch.stack([layer(xv) for xv in x])
    x = x.transpose(0, 1).contiguous()  # B V
    return x