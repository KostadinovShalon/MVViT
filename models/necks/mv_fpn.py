# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from MVViT.models.utils.multi_view_layers import apply_multiview_layer
from mmdet.models import NECKS, FPN


@NECKS.register_module()
class MVFPN(FPN):

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            apply_multiview_layer(inputs[i + self.start_level], lateral_conv)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + apply_multiview_layer(laterals[i],
                                                                          F.interpolate, **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[3:]
                laterals[i - 1] = laterals[i - 1] + apply_multiview_layer(laterals[i],
                                                                          F.interpolate,
                                                                          size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            apply_multiview_layer(laterals[i], self.fpn_convs[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(apply_multiview_layer(outs[-1], F.max_pool2d, 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(apply_multiview_layer(extra_source, self.fpn_convs[used_backbone_levels]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(apply_multiview_layer(F.relu(outs[-1]), self.fpn_convs[i]))
                    else:
                        outs.append(apply_multiview_layer(outs[-1], self.fpn_convs[i]))
        return tuple(outs)
