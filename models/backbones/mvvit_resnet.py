import torch

from MVViT.models.backbones.mvvit_darknet import MultiViewPositionalEncoding, MVTransformer
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
                 views=2,
                 multiview_decoder_mode='add',
                 shared_transformer=False
                 ):
        super(MVViTResNet, self).__init__(depth, in_channels, stem_channels, base_channels, num_stages, strides,
                                          dilations, out_indices, style, deep_stem, avg_down, frozen_stages,
                                          conv_cfg, norm_cfg, norm_eval, dcn, stage_with_dcn, plugins, with_cp,
                                          zero_init_residual, pretrained, init_cfg)

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        self.shared_transformer = shared_transformer

        # Obtaining sampling points
        assert max(self.combination_blocks) <= num_stages + 1
        self.grid_sizes = []
        self.positional_encoding = {} if positional_encoding else None
        self.transformers = {}
        self.views = views
        for cb in self.combination_blocks:
            # fh, fw = input_size
            # for _ in range(cb):
            #     fw = (fw + 1) // 2
            #     fh = (fh + 1) // 2
            # self.grid_sizes.append((fh, fw))
            if cb > -1:  # cb == 0 means combine before the first layer, < 0 means no combine
                if cb == 0:
                    fus_ft_dims = 3
                elif cb == 1:
                    fus_ft_dims = self.base_channels
                else:
                    fus_ft_dims = (self.base_channels * 2 ** (cb - 2)) * self.block.expansion

                if positional_encoding:
                    self.positional_encoding[cb] = MultiViewPositionalEncoding(fus_ft_dims)
                if not shared_transformer:
                    view_transformers = []
                    for v in range(views):
                        layer_name = f'transformer_c{cb}_v{v}'
                        self.add_module(
                            layer_name,
                            MVTransformer(fus_ft_dims, nhead, num_decoder_layers, n_views=views,
                                          mode=multiview_decoder_mode))
                        view_transformers.append(layer_name)
                else:
                    view_transformers = f'transformer_c{cb}'
                    self.add_module(
                        view_transformers,
                        MVTransformer(fus_ft_dims, nhead, num_decoder_layers,
                                      n_views=views, mode=multiview_decoder_mode))
                self.transformers[cb] = view_transformers

    def forward(self, x, with_attn_weights=False):
        """Forward function."""
        # x is now B x V x C x W x H
        if 0 in self.combination_blocks:
            x, attn = self.mv_transformer(x, 0, with_attn_weights)
        if self.deep_stem:
            x = self.apply_multiview_layer(x, self.stem)
        else:
            x = self.apply_multiview_layer(x, [self.conv1, self.norm1, self.relu])
        x = self.apply_multiview_layer(x, self.maxpool)

        outs = []

        for i, layer_name in enumerate(self.res_layers):
            if (i + 1) in self.combination_blocks:
                x, attn = self.mv_transformer(x, i + 1, with_attn_weights)
            res_layer = getattr(self, layer_name)
            x = self.apply_multiview_layer(x, res_layer)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def mv_transformer(self, x, comb_block, with_attn_weights=False):
        if self.positional_encoding is not None:
            x = self.positional_encoding[comb_block](x)

        x = x.transpose(0, 1)  # V B
        outputs = []
        attn_weights = []
        for v in range(self.views):
            other_views = [xv.permute(0, 2, 3, 1) for i, xv in enumerate(x) if i != v]
            layer_name = self.transformers[comb_block] if self.shared_transformer else self.transformers[comb_block][v]
            t = getattr(self, layer_name)
            output, wts = t(other_views, x[v].permute(0, 2, 3, 1))  # a "n_dec" list of "v-1" list of tensors
            outputs.append(output.permute(0, 3, 1, 2))  # B x C x H x W
            if with_attn_weights:
                attn_weights.append(torch.stack(wts[0]))  # (V-1) x B x H x W x H x W TODO: change to all attns

        outputs = torch.stack(outputs).transpose(0, 1).contiguous()  # B x V x C x W x H
        if with_attn_weights:
            return outputs, torch.stack(attn_weights).permute(2, 0, 1, 3, 4, 5,
                                                              6).contiguous()  # B x V x (V-1) x H x W x H x W
        return outputs, None

    @staticmethod
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
