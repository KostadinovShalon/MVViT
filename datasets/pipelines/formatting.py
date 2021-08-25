from collections.abc import Sequence

import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import to_tensor, ImageToTensor
import torch


@PIPELINES.register_module()
class MVFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            imgs = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            imgs = tuple(np.expand_dims(img, -1) if len(img.shape) < 3 else img for img in imgs)
            imgs = tuple(np.ascontiguousarray(img.transpose(2, 0, 1)) for img in imgs)
            imgs = torch.stack([to_tensor(img) for img in imgs])
            results['img'] = DC(imgs, stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC([to_tensor(r) for r in results[key]])
        if 'gt_masks' in results:
            results['gt_masks'] = DC(torch.stack([to_tensor(r) for r in results['gt_masks']]), cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(torch.cat(
                [to_tensor(r[None, ...]) for r in results['gt_semantic_seg']]), stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        imgs = results['img']
        results.setdefault('pad_shape', tuple(img.shape for img in imgs))
        results.setdefault('scale_factor', tuple([1.0] * len(imgs)))
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class MVImageToTensor(ImageToTensor):

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            imgs = tuple(np.expand_dims(img, -1) if len(img.shape) < 3 else img for img in results[key])
            results[key] = torch.stack([to_tensor(img.transpose(2, 0, 1)) for img in imgs])
        return results
