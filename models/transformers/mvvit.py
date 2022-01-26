import torch
from torch import nn

from MVViT.models.transformers.mv_transformer import MVTransformer, MultiViewPositionalEncoding


class MVViT(nn.Module):

    def __init__(self, feature_dim, nhead=8, num_decoder_layers=1, positional_encoding=True, views=2):
        super(MVViT, self).__init__()
        self.transformer = MVTransformer(feature_dim, nhead, num_decoder_layers, n_views=views)
        self.positional_encoding = MultiViewPositionalEncoding(feature_dim) if positional_encoding else None

    def __call__(self, x, with_attn_weights=False):
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        x = x.transpose(0, 1)  # V B
        outputs = []
        attn_weights = []
        for v in range(self.views):
            other_views = [xv.permute(0, 2, 3, 1) for i, xv in enumerate(x) if i != v]
            output, wts = self.transformer(other_views, x[v].permute(0, 2, 3, 1),
                                           need_attn_weights=with_attn_weights)  # a "n_dec" list of v-1 list of tensors
            outputs.append(output.permute(0, 3, 1, 2))  # B x C x H x W
            if with_attn_weights:
                attn_weights.append(wts[0])  # B x H x W x H x W

        outputs = torch.stack(outputs).transpose(0, 1).contiguous()  # B x V x C x W x H
        if with_attn_weights:
            attn_weights = torch.stack(attn_weights).permute(1, 0, 2, 3, 4,
                                                             5).contiguous()  # B x V x H x W x H x W
            return outputs, attn_weights
        return outputs, None
