from MVViT.models.transformers.mvvit import MVViT
from MVViT.models.utils.multi_view_layers import apply_multiview_patch_embed_layer, apply_multiview_layer, \
    apply_multiview_swin_sequence_layer
from mmdet.models import BACKBONES, SwinTransformer


@BACKBONES.register_module()
class MVViTSwinTransformer(SwinTransformer):
    def __init__(self, pretrain_img_size=224, in_channels=3, embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3),
                 qkv_bias=True, qk_scale=None, patch_norm=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 use_abs_pos_embed=False, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False,
                 pretrained=None, convert_weights=False, frozen_stages=-1, init_cfg=None, combination_block=-1, nhead=8,
                 num_decoder_layers=1, positional_encoding=True, views=2):

        super().__init__(pretrain_img_size, in_channels, embed_dims, patch_size, window_size, mlp_ratio, depths,
                         num_heads, strides, out_indices, qkv_bias, qk_scale, patch_norm, drop_rate, attn_drop_rate,
                         drop_path_rate, use_abs_pos_embed, act_cfg, norm_cfg, with_cp, pretrained, convert_weights,
                         frozen_stages, init_cfg)

        num_layers = len(depths)

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        assert max(self.combination_blocks) <= num_layers + 1
        self.mvvits = {}
        self.views = views
        for cb in self.combination_blocks:
            if cb > -1:  # cb == 0 means combine before the first layer, < 0 means no combine
                if cb == 0:
                    fus_ft_dims = 3
                elif cb == 1:  # cb == 1 means after the patch embed but before the swin blocks
                    fus_ft_dims = embed_dims
                elif cb - 2 < num_layers - 1:
                    fus_ft_dims = 2 ** (cb - 1) * embed_dims
                else:  # cb - 2 == num_layers
                    fus_ft_dims = 2 ** (num_layers - 1) * embed_dims

                view_mvvit = f'mvvit_c{cb}'
                self.add_module(
                    view_mvvit,
                    MVViT(fus_ft_dims, nhead=nhead,
                          num_decoder_layers=num_decoder_layers,
                          positional_encoding=positional_encoding,
                          views=views))
                self.mvvits[cb] = view_mvvit

    def forward(self, x, with_attn_weights=False):
        # x is B x V x C x W x H

        if 0 in self.combination_blocks:
            layer_name = self.mvvits[0]
            mvvit = getattr(self, layer_name)
            x, attn = mvvit(x, with_attn_weights)

        x, hw_shape = apply_multiview_patch_embed_layer(x, self.patch_embed)

        if self.use_abs_pos_embed:
            x = apply_multiview_layer(x, lambda y: y + self.absolute_pos_embed)
        x = apply_multiview_layer(x, self.drop_after_pos)

        if 1 in self.combination_blocks:
            layer_name = self.mvvits[1]
            mvvit = getattr(self, layer_name)
            x, attn = mvvit(x, with_attn_weights)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = apply_multiview_swin_sequence_layer(x, hw_shape, stage)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = apply_multiview_layer(out, norm_layer)
                out = out.view(-1, self.views, *out_hw_shape,
                               self.num_features[i]).permute(0, 1, 4, 2,
                                                             3).contiguous()
                outs.append(out)
            if i + 2 in self.combination_blocks:
                mvvit_layer_name = self.mvvits[i + 2]
                mvvit = getattr(self, mvvit_layer_name)
                b, v, n, c = x.shape
                x = x.transpose(2, 3).view(b, v, c, *hw_shape)
                x, attn = mvvit(x, with_attn_weights)
                x = x.view(b, v, c, n).transpose(2, 3)
        outs = [out.view(-1, *out.shape[2:]) for out in outs]  # (b . v) x f x w' x h' ordered per view
        if with_attn_weights:
            return tuple(outs), attn
        return tuple(outs)
