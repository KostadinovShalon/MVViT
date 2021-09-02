import torch

from mmdet.models.utils import Transformer
from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER.register_module()
class MVTransformer(Transformer):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """
    def __init__(self, encoder=None, decoder=None, init_cfg=None, single_view=False):
        super(MVTransformer, self).__init__(encoder, decoder, init_cfg)
        self.single_view = single_view

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, v, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, v, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs, views,
                      num_query, embed_dims], else has shape [1, bs, views, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, v, embed_dims, h, w].
        """
        bs, v, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, v, c, -1).permute(1, 3, 0, 2)  # [bs, v, c, h, w] -> [v, h*w, bs, c]
        pos_embed = pos_embed.view(bs, v, c, -1).permute(1, 3, 0, 2)
        query_embed = query_embed.unsqueeze(1).unsqueeze(0).repeat(
            1, bs, 1)  # [num_query, dim] -> [v, num_query, bs, dim]
        mask = mask.view(bs, v, -1)  # [bs, v, h, w] -> [bs, v, h*w]

        memory = torch.stack([self.encoder(
            query=x[_v],
            key=None,
            value=None,
            query_pos=pos_embed[_v],
            query_key_padding_mask=mask[:, _v, :]) for _v in range(v)])
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, views, num_query, bs, dim]
        if self.single_view:
            out_dec = torch.stack([self.decoder(
                query=target,
                key=memory[_v],
                value=memory[_v],
                key_pos=pos_embed[_v],
                query_pos=query_embed,
                key_padding_mask=mask[:, _v, :]) for _v in range(v)], dim=1)
        else:
            appended_memory = memory.view(-1, bs, c)
            appended_pos_embed = pos_embed.view(-1, bs, c)
            appended_mask = mask.view(bs, -1)
            out_dec = self.decoder(
                query=target,
                key=appended_memory,
                value=appended_memory,
                key_pos=appended_pos_embed,
                query_pos=query_embed,
                key_padding_mask=appended_mask)
            num_layers = out_dec.shape[0]
            out_dec = out_dec.view(num_layers, v, -1, bs, c)
        out_dec = out_dec.permute(3, 1, 2)
        memory = memory.permute(2, 0, 3, 1).reshape(bs, v, c, h, w)
        return out_dec, memory
