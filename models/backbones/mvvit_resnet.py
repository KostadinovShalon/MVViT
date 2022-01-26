from MVViT.models.transformers.mvvit import MVViT
from MVViT.models.utils.multi_view_layers import apply_multiview_layer
from mmdet.models import ResNet

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class MVViTResNet(ResNet):

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 combination_block=-1,
                 nhead=8,
                 num_decoder_layers=1,
                 positional_encoding=True,
                 views=2
                 ):
        super(MVViTResNet, self).__init__(depth, in_channels, stem_channels, base_channels, num_stages, strides,
                                          dilations, out_indices, style, deep_stem, avg_down, frozen_stages,
                                          conv_cfg, norm_cfg, norm_eval, dcn, stage_with_dcn, plugins, with_cp,
                                          zero_init_residual, pretrained, init_cfg)

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        assert max(self.combination_blocks) <= num_stages + 1
        self.mvvits = {}
        self.views = views
        for cb in self.combination_blocks:
            if cb > -1:  # cb == 0 means combine before the first layer, < 0 means no combine
                if cb == 0:
                    fus_ft_dims = 3
                elif cb == 1:
                    fus_ft_dims = self.base_channels
                else:
                    fus_ft_dims = (self.base_channels * 2 ** (cb - 2)) * self.block.expansion

                view_mvvit = f'mvvit_c{cb}'
                self.add_module(
                    view_mvvit,
                    MVViT(fus_ft_dims, nhead=nhead,
                          num_decoder_layers=num_decoder_layers,
                          positional_encoding=positional_encoding,
                          views=views))
                self.view_mvvit[cb] = view_mvvit

    def forward(self, x, with_attn_weights=False):
        """Forward function."""
        # x is now B x V x C x W x H
        if 0 in self.combination_blocks:
            layer_name = self.mvvits[0]
            mvvit = getattr(self, layer_name)
            x, attn = mvvit(x, with_attn_weights)
        if self.deep_stem:
            x = apply_multiview_layer(x, self.stem)
        else:
            x = apply_multiview_layer(x, [self.conv1, self.norm1, self.relu])
        x = apply_multiview_layer(x, self.maxpool)

        outs = []

        for i, layer_name in enumerate(self.res_layers):
            if (i + 1) in self.combination_blocks:
                layer_name = self.mvvits[i + 1]
                mvvit = getattr(self, layer_name)
                x, attn = mvvit(x, with_attn_weights)
            res_layer = getattr(self, layer_name)
            x = apply_multiview_layer(x, res_layer)
            if i in self.out_indices:
                outs.append(x)
        if with_attn_weights:
            return tuple(outs), attn
        return tuple(outs)
