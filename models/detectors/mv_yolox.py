# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet.models.detectors.yolox import YOLOX


@DETECTORS.register_module()
class MVYOLOX(YOLOX):
    def _preprocess(self, img, gt_bboxes):
        # img is a B V C H W and gt_bboxes is b-list of v-list
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img_shape = img.shape
            img = img.flatten(0, 1)
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            new_img_shape = img.shape
            img = img.view(*img_shape[:3], *new_img_shape[-2:])
            for v_gt_bboxes in gt_bboxes:
                for gt_bbox in v_gt_bboxes:
                    gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                    gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes
