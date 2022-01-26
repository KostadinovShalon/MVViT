# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from MVViT.models.transformers.mvvit import MVViT
from MVViT.models.utils.multi_view_layers import apply_multiview_layer
from mmdet.models.backbones.darknet import Darknet
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class MVViTDarknet(Darknet):
    """Darknet backbone with multi-view integration at the full view.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.

    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = MVViTDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True,
                 init_cfg=None,
                 combination_block=4,
                 nhead=8,
                 num_decoder_layers=1,
                 positional_encoding=True,
                 views=2):
        super(MVViTDarknet, self).__init__(depth, out_indices, frozen_stages, conv_cfg, norm_cfg,
                                           act_cfg, norm_eval, None, init_cfg)

        self.views = views

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        assert max(self.combination_blocks) <= len(self.layers)

        self.mvvits = {}
        for cb in self.combination_blocks:
            if cb > -2:  # cb == -1 means combine before the first layer, < -1 means no combine
                if cb == len(self.channels):
                    fus_ft_dims = self.channels[cb - 1][1]
                else:
                    fus_ft_dims = 3 if cb < 0 else self.channels[cb][0]

                view_mvvit = f'mvvit_c{cb}'
                self.add_module(
                    view_mvvit,
                    MVViT(fus_ft_dims, nhead=nhead,
                          num_decoder_layers=num_decoder_layers,
                          positional_encoding=positional_encoding,
                          views=views))
                self.mvvits[cb] = view_mvvit

    def forward(self, x, with_attn_weights=False):
        # x = x.permute(1, 0, 2, 3, 4)  # x is now V x B x C x W x H
        outs = []
        attn = None
        if -1 in self.combination_blocks:
            layer_name = self.mvvits[-1]
            mvvit = getattr(self, layer_name)
            x, attn = mvvit(x, with_attn_weights)
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = apply_multiview_layer(x, cr_block)
            if i in self.combination_blocks:
                layer_name = self.mvvits[i]
                mvvit = getattr(self, layer_name)
                x, attn = mvvit(x, with_attn_weights)
            if i in self.out_indices:
                outs.append(x)

        outs = [out.view(-1, *out.shape[2:]) for out in outs]  # (b . v) x f x w' x h' ordered per view

        if with_attn_weights:
            return tuple(outs), attn
        return tuple(outs)

