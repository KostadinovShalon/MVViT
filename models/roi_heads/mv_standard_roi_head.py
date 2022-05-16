# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_sampler
from mmdet.models import StandardRoIHead, HEADS


@HEADS.register_module()
class MVStandardRoIHead(StandardRoIHead):

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals

        img_metas = [
            {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
            img_meta in
            img_metas for v in range(len(img_meta['img_shape']))]
        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes for v_b_gt_bboxes in b_gt_bboxes]
        if gt_labels is not None:
            gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels for v_b_gt_labels in b_gt_labels]

        return super().forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks, **kwargs)

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        img_metas = [
            {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
            img_meta in
            img_metas for v in range(len(img_meta['img_shape']))]
        return super().async_simple_test(x, proposal_list, img_metas, proposals, rescale)

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        img_metas = [
            {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
            img_meta in
            img_metas for v in range(len(img_meta['img_shape']))]
        return super().simple_test(x, proposal_list, img_metas, proposals, rescale)

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        img_metas = [
            {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
            img_meta in
            img_metas for v in range(len(img_meta['img_shape']))]
        return super().aug_test(x, proposal_list, img_metas, rescale)
