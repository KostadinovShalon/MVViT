# Copyright (c) OpenMMLab. All rights reserved.
import math

from MVViT.models.transformers.mvvit import MVViT
from MVViT.models.utils.multi_view_layers import apply_multiview_layer
from mmdet.models import CSPDarknet, BACKBONES


@BACKBONES.register_module()
class MVViTCSPDarknet(CSPDarknet):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
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

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # arch_settings = {
    #     'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
    #            [256, 512, 9, True, False], [512, 1024, 3, False, True]],
    #     'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
    #            [256, 512, 9, True, False], [512, 768, 3, True, False],
    #            [768, 1024, 3, False, True]]
    # }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 combination_block=4,
                 nhead=8,
                 num_decoder_layers=1,
                 positional_encoding=True,
                 views=2):
        super(MVViTCSPDarknet, self).__init__(arch, deepen_factor, widen_factor, out_indices, frozen_stages,
                                              use_depthwise, arch_ovewrite, spp_kernal_sizes, conv_cfg,
                                              norm_cfg, act_cfg, norm_eval, init_cfg)

        self.views = views

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        assert max(self.combination_blocks) <= 6
        arch_setting = self.arch_settings[arch]

        self.mvvits = {}
        for cb in self.combination_blocks:
            if cb >= 0:  # cb == 0 means combine before the first layer, < 0 means no combine
                if cb == 0:
                    fus_ft_dims = 3
                else:
                    fus_ft_dims = int(arch_setting[cb - 1][0] * widen_factor)

                view_mvvit = f'mvvit_c{cb}'
                self.add_module(
                    view_mvvit,
                    MVViT(fus_ft_dims, nhead=nhead,
                          num_decoder_layers=num_decoder_layers,
                          positional_encoding=positional_encoding,
                          views=views))
                self.mvvits[cb] = view_mvvit

    def forward(self, x, with_attn_weights=False):
        outs = []
        attn = None
        for i, layer_name in enumerate(self.layers):

            if i in self.combination_blocks:
                mvvit_layer_name = self.mvvits[i]
                mvvit = getattr(self, mvvit_layer_name)
                x, attn = mvvit(x, with_attn_weights)

            layer = getattr(self, layer_name)
            x = apply_multiview_layer(x, layer)
            if i in self.out_indices:
                outs.append(x)
        outs = [out.view(-1, *out.shape[2:]) for out in outs]  # (b . v) x f x w' x h' ordered per view

        if with_attn_weights:
            return tuple(outs), attn
        return tuple(outs)
