# Copyright (c) 2019 Western Digital Corporation or its affiliates.
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint, BaseModule
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.linear import Linear

from mmdet.models.builder import BACKBONES


class ResBlock(nn.Module):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, half_in_channels, 1, **cfg)
        self.conv2 = ConvModule(
            half_in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


class MultiViewPositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super(MultiViewPositionalEncoding, self).__init__()

        channels = int(np.ceil(d_model / 6) * 2)
        self.channels = channels

        self.div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))

    def forward(self, x):
        # x is B x C x W x H
        b, v, c, h, w = x.shape
        position_x = torch.arange(w, dtype=torch.float).unsqueeze(1)  # W x 1
        position_y = torch.arange(h, dtype=torch.float).unsqueeze(1)  # H x 1
        position_view = torch.arange(v, dtype=torch.float).unsqueeze(1)  # V x 1

        pe = torch.zeros(v, self.channels * 3, h, w)
        pe[:, 0:self.channels:2, ...] = torch.sin(position_x * self.div_term).transpose(0, 1).unsqueeze(1).unsqueeze(
            0).repeat(v, 1, h, 1)
        pe[:, 1:self.channels:2, ...] = torch.cos(position_x * self.div_term).transpose(0, 1).unsqueeze(1).unsqueeze(
            0).repeat(v, 1, h, 1)
        pe[:, self.channels:(2 * self.channels):2, ...] = torch.sin(position_y * self.div_term).transpose(0,
                                                                                                          1).unsqueeze(
            2).unsqueeze(
            0).repeat(v, 1, 1, w)
        pe[:, (self.channels + 1):(2 * self.channels):2, ...] = torch.cos(position_y * self.div_term).transpose(0,
                                                                                                                1).unsqueeze(
            2).unsqueeze(
            0).repeat(v, 1, 1, w)
        pe[:, (2 * self.channels)::2, ...] = torch.sin(position_view * self.div_term).unsqueeze(
            2).unsqueeze(
            2).repeat(1, 1, h, w)
        pe[:, (2 * self.channels + 1)::2, ...] = torch.cos(position_view * self.div_term).unsqueeze(
            2).unsqueeze(
            2).repeat(1, 1, h, w)

        pe = pe.repeat(b, 1, 1, 1, 1).to(x.device)
        x = x + pe[:, :, :c, ...]
        return x


class MultiheadAttentionND(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttentionND, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionND, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        need_weights: output attn_output_weights.
        key_padding_mask: TODO not implemented.
        attn_mask: TODO not implemented

    Shapes for inputs:
        - query: :math:`(N, *, E)` where B is the batch size, * is any number of additional dimensions, E is
          the embedding dimension.
        - key: :math:`(N, *, E)`, where B is the batch size, * is any number of additional dimensions, E is
          the embedding dimension.
        - value: :math:`(N, *, E)` where B is the batch size, * is any number of additional dimensions, E is
          the embedding dimension.

    Shapes for outputs:
        - attn_output: :math:`(N, *, E)` where B is the batch size, * is any number of additional dimensions, E is
          the embedding dimension.
        - attn_output_weights: :math:`(N, *q, *kv)` where B is the batch size, *q is any number of additional dimensions
          of the query and *kv are the additional dimensions of the key-value pairs
        """

        bsz, embed_dim = query.size(0), query.size(-1)
        assert embed_dim == self.embed_dim
        # allow MHA to have different sizes for the feature dimension
        q_extra_dims = query.shape[1:-1]
        k_extra_dims = key.shape[1:-1]
        assert key.shape[1:-1] == value.shape[1:-1]

        scaling = float(self.head_dim) ** -0.5

        if self._qkv_same_embed_dim:
            if torch.equal(query, key) and torch.equal(key, value):
                # self-attention
                q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

            elif torch.equal(key, value):
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = F.linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = self.in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = self.in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = self.linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = self.linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = F.linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(self.q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(self.k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(self.v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if self.in_proj_bias is not None:
                q = F.linear(query, q_proj_weight_non_opt, self.in_proj_bias[0:embed_dim])
                k = F.linear(key, k_proj_weight_non_opt, self.in_proj_bias[embed_dim:(embed_dim * 2)])
                v = F.linear(value, v_proj_weight_non_opt, self.in_proj_bias[(embed_dim * 2):])
            else:
                q = F.linear(query, q_proj_weight_non_opt, self.in_proj_bias)
                k = F.linear(key, k_proj_weight_non_opt, self.in_proj_bias)
                v = F.linear(value, v_proj_weight_non_opt, self.in_proj_bias)
        q = q * scaling

        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(bsz, *([1] * (len(k.shape) - 1)))])
            v = torch.cat([v, self.bias_v.repeat(bsz, *([1] * (len(v.shape) - 1)))])
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        q = q.contiguous().view(bsz, -1, self.embed_dim).transpose(0, 1).contiguous() \
            .view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # B*H, ..., d_h
        if k is not None:
            k = k.contiguous().view(bsz, -1, self.embed_dim).transpose(0, 1).contiguous() \
                .view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(bsz, -1, self.embed_dim).transpose(0, 1).contiguous() \
                .view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.matmul(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, np.prod(q_extra_dims),
                                                    np.prod(k_extra_dims)]

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, np.prod(q_extra_dims), self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim) \
            .transpose(0, 1).view(bsz, *q_extra_dims, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, *q_extra_dims,
                                                           *k_extra_dims)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class MVTransformer(nn.Transformer):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 n_views=2,
                 mode='add') -> None:
        decoder_layer = MVTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  other_views=n_views - 1, mode=mode)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.mode = mode
        super(MVTransformer, self).__init__(d_model, nhead, num_decoder_layers, num_decoder_layers, dim_feedforward,
                                            dropout, activation, custom_decoder=decoder)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src[0].size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        # if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = src
        if self.mode == 'cat':
            memory = torch.cat(memory, dim=-1)
        output, attn_weights = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
        return output, attn_weights


class TransformerDecoder(nn.TransformerDecoder):

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        attn_weights_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
            attn_weights_list.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights_list  # attn_weights_list is a list of "B x H x W x H x W tensors"


class MVTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", other_views=1, mode='add'):
        super(MVTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = MultiheadAttentionND(d_model, nhead, dropout=dropout)
        if mode == 'add':
            self.multihead_attn = nn.ModuleList()
            for v in range(other_views):
                self.multihead_attn.append(MultiheadAttentionND(d_model, nhead, dropout=dropout,
                                                                kdim=d_model, vdim=d_model))
        elif mode == 'cat':
            self.multihead_attn = MultiheadAttentionND(d_model, nhead, dropout=dropout,
                                                       kdim=d_model * other_views, vdim=d_model * other_views)
        else:
            raise NotImplemented(f"Decoder mode {mode} has not been implemented.")
        self.mode = mode
        self.other_views = other_views

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.mode == "add":
            tgt2, attn_weights = [], []
            for v in range(self.other_views):
                _tgt2, _attn_weights = self.multihead_attn[v](tgt, memory[v], memory[v], attn_mask=memory_mask,
                                                              key_padding_mask=memory_key_padding_mask,
                                                              need_weights=True)
                tgt2.append(_tgt2)
                attn_weights.append(_attn_weights)
            tgt2 = torch.stack(tgt2).sum(dim=0)
        else:
            tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask,
                                                     need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights


@BACKBONES.register_module()
class MVViTDarknet(BaseModule):
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

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True,
                 init_cfg=None,
                 input_size=(608, 608),
                 combination_block=4,
                 nhead=8,
                 num_decoder_layers=1,
                 positional_encoding=True,
                 views=2,
                 multiview_decoder_mode='add',
                 shared_transformer=False):
        super(MVViTDarknet, self).__init__(init_cfg)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]
        self.views = views

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, **cfg))
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval

        if not isinstance(combination_block, list):
            combination_block = [combination_block]

        self.combination_blocks = combination_block

        self.shared_transformer = shared_transformer

        # Obtaining sampling points
        assert max(self.combination_blocks) <= len(self.layers)
        self.grid_sizes = []
        self.positional_encoding = {} if positional_encoding else None
        self.transformers = {}
        for cb in self.combination_blocks:
            fh, fw = input_size
            for _ in range(cb):
                fw = (fw + 1) // 2
                fh = (fh + 1) // 2
            self.grid_sizes.append((fh, fw))
            if cb > -2:  # cb == -1 means combine before the first layer, < -1 means no combine
                if cb == len(self.channels):
                    fus_ft_dims = self.channels[cb - 1][1]
                else:
                    fus_ft_dims = 3 if cb < 0 else self.channels[cb][0]

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
        # x = x.permute(1, 0, 2, 3, 4)  # x is now V x B x C x W x H
        outs = []
        attn = None
        if -1 in self.combination_blocks:
            x, attn = self.mv_transformer(x, -1, with_attn_weights)
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = x.transpose(0, 1).contiguous()  # V B
            x = torch.stack([cr_block(xv) for xv in x])
            x = x.transpose(0, 1).contiguous()  # B V
            if i in self.combination_blocks:
                x, attn = self.mv_transformer(x, i, with_attn_weights)
            if i in self.out_indices:
                outs.append(x)

        outs = [out.view(-1, *out.shape[2:]) for out in outs]  # (v . b) x f x w' x h' ordered per view

        if with_attn_weights:
            return tuple(outs), attn
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

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MVViTDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx),
                             ResBlock(out_channels, **cfg))
        return model
