from functools import partial

import torch
from timm.models.layers import PatchEmbed, trunc_normal_, DropPath, Mlp
from timm.models.layers.helpers import to_2tuple
from torch import nn
from timm.models.vision_transformer import Block, _init_vit_weights, Attention


class GeneralAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def to_heads(self, x, linear_layer):
        B, N, C = x.shape
        x = linear_layer(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        return x

    def forward(self, q, k, v):
        B, N, C = q.shape
        q = self.to_heads(q, self.q_lin)  # B x H x N x C
        k = self.to_heads(k, self.k_lin)
        v = self.to_heads(v, self.v_lin)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x H x Nq x Nk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B x H x Nq x C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mem_norm = norm_layer(dim)
        self.attn = GeneralAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory):
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        memory = self.mem_norm(memory)
        x = x + self.drop_path(self.attn(self.norm2(x), memory, memory))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class MVVisionTransformerEncoder(nn.Module):

    def __init__(self, img_size=544, patch_size=16, in_chans=3, out_chans=512, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        output_chans = out_chans if out_chans is not None else in_chans
        self.head = nn.Linear(self.num_features, output_chans)

        # Weight init
        assert weight_init in ('jax', '')
        trunc_normal_(self.pos_embed, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, jax_impl=True)
        else:
            self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # B x N x C

    def forward(self, x):
        x = self.forward_features(x)
        # B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, *self.grid_size)
        x = self.head(x)
        return x


class MVVisionTransformerDecoder(nn.Module):

    def __init__(self, img_size=34, patch_size=1, in_chans=512, out_chans=None, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        output_chans = out_chans if out_chans is not None else in_chans
        self.head = nn.Conv2d(self.num_features, output_chans, kernel_size=1)

        # Weight init
        assert weight_init in ('jax', '')
        trunc_normal_(self.pos_embed, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, jax_impl=True)
        else:
            self.apply(_init_vit_weights)

    def forward_features(self, x, memory):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, memory)
        x = self.norm(x)
        return x  # B x N x C

    def forward(self, x, memory):
        x = self.forward_features(x, memory)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, *self.grid_size)  # B x C x Gx x Gy
        x = self.head(x)
        return x


class MVTransformer(nn.Module):

    def __init__(self, img_size=544, feature_size=34, patch_size=16, in_chans=3, feature_channels=512,
                 encoder_embed_dim=768, decoder_embed_dim=768, encoder_depth=6, decoder_depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.encoder = MVVisionTransformerEncoder(img_size, patch_size, in_chans, decoder_embed_dim,
                                                  encoder_embed_dim, encoder_depth,
                                                  num_heads, mlp_ratio, qkv_bias, qk_scale,
                                                  drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                                                  act_layer, weight_init)
        # self.decoder = MVVisionTransformerDecoder(img_size, patch_size, in_chans, None,
        #                                           decoder_embed_dim, decoder_depth,
        #                                           num_heads, mlp_ratio, qkv_bias, qk_scale,
        #                                           drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
        #                                           act_layer, weight_init)#
        # self.img_size = to_2tuple(img_size)
        self.feature_size = to_2tuple(feature_size)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = embed_layer(
            img_size=feature_size, patch_size=1, in_chans=feature_channels, embed_dim=decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm = norm_layer(decoder_embed_dim)
        self.norm2 = norm_layer(decoder_embed_dim)
        self.attn = GeneralAttention(
            decoder_embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.output_layer = nn.Conv2d(decoder_embed_dim, feature_channels, kernel_size=1)

    def forward(self, x, other_view_ftrs):
        other_view = self.encoder(other_view_ftrs)

        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = x + self.drop_path(self.attn(self.norm(x), other_view, other_view))
        x = self.norm2(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, *self.feature_size)
        x = self.output_layer(x)
        return x
