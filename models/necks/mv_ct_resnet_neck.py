# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16

from MVViT.models.backbones.mvvit_darknet import MultiViewPositionalEncoding, MVTransformer
from mmdet.models import CTResNetNeck
from mmdet.models.builder import NECKS


@NECKS.register_module()
class MVCTResNetNeck(CTResNetNeck):
    """The (MV) neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 use_dcn=True,
                 init_cfg=None,
                 positional_encoding=True,
                 views=2,
                 nhead=8,
                 num_decoder_layers=1,
                 multiview_decoder_mode='add',
                 single_view=False):
        super(MVCTResNetNeck, self).__init__(in_channel, num_deconv_filters, num_deconv_kernels, use_dcn, init_cfg)
        self.positional_encoding = MultiViewPositionalEncoding(in_channel) if positional_encoding else nn.Identity()
        self.views = views
        self.transformer = MVTransformer(in_channel, nhead, num_decoder_layers, n_views=views,
                                         mode=multiview_decoder_mode)
        self.single_view = single_view

    @auto_fp16()
    def forward(self, inputs, with_attn_weights=False):
        assert isinstance(inputs, (list, tuple))
        x = inputs[-1]
        if not self.single_view:
            x, _ = self.mv_transformer(x, with_attn_weights)
        x = x.transpose(0, 1).contiguous()
        x = torch.stack([self.deconv_layers(xv) for xv in x])
        x = x.transpose(0, 1).contiguous()  # B V
        x = x.view(-1, *x.shape[2:])  # (b.v) x f x w' x h' ordered per view
        return x,

    def init_weights(self):
        super().init_weights()
        self.transformer.init_weights()

    def mv_transformer(self, x, with_attn_weights=False):
        x = self.positional_encoding(x)

        x = x.transpose(0, 1)  # V B
        outputs = []
        attn_weights = []
        for v in range(self.views):
            other_views = [xv.permute(0, 2, 3, 1) for i, xv in enumerate(x) if i != v]
            output, wts = self.transformer(other_views, x[v].permute(0, 2, 3, 1))
            outputs.append(output.permute(0, 3, 1, 2))  # B x C x H x W
            if with_attn_weights:
                attn_weights.append(torch.stack(wts[0]))  # (V-1) x B x H x W x H x W TODO: change to all attns

        outputs = torch.stack(outputs).transpose(0, 1).contiguous()  # B x V x C x W x H
        if with_attn_weights:
            return outputs, torch.stack(attn_weights).permute(2, 0, 1, 3, 4, 5,
                                                              6).contiguous()  # B x V x (V-1) x H x W x H x W
        return outputs, None
