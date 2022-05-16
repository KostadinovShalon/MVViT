# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import images_to_levels, multi_apply, select_single_mlvl
from mmdet.models import RPNHead, HEADS


@HEADS.register_module()
class MVRPNHead(RPNHead):

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes for v_b_gt_bboxes in b_gt_bboxes]
        # if gt_labels is not None:
        #     gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels for v_b_gt_labels in b_gt_labels]
        img_metas = [
            {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
            img_meta in
            img_metas for v in range(len(img_meta['img_shape']))]
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0][0].dtype,
            device=cls_scores[0][0].device)
        views = len(img_metas[0]['img_shape'])
        result_list = []

        # img_metas = [
        #     {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()} for
        #     img_meta in
        #     img_metas for v in range(views)]
        for v in range(views):
            v_cls_scores = [
                cls_scores[i][v::views].detach() for i in range(num_levels)
            ]
            v_bbox_preds = [
                bbox_preds[i][v::views].detach() for i in range(num_levels)
            ]
            v_score_factors = None
            if score_factors is not None:
                v_score_factors = [
                    score_factors[i][v::views].detach() for i in range(num_levels)
                ]
            v_img_metas = [
                {k: (val[v] if k not in ("img_norm_cfg", "batch_input_shape") else val) for k, val in img_meta.items()}
                for img_meta in img_metas]
            for img_id in range(len(v_img_metas)):
                img_meta = v_img_metas[img_id]
                cls_score_list = select_single_mlvl(v_cls_scores, img_id)
                bbox_pred_list = select_single_mlvl(v_bbox_preds, img_id)
                if with_score_factors:
                    score_factor_list = select_single_mlvl(v_score_factors, img_id)
                else:
                    score_factor_list = [None for _ in range(num_levels)]

                results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                  score_factor_list, mlvl_priors,
                                                  img_meta, cfg, rescale, with_nms,
                                                  **kwargs)
                result_list.append(results)
        return result_list


    # def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
    #                        level_ids, cfg, img_shape, **kwargs):
    #     """bbox post-processing method.
    #
    #     Do the nms operation for bboxes in same level.
    #
    #     Args:
    #         mlvl_scores (list[Tensor]): Box scores from all scale
    #             levels of a single image, each item has shape
    #             (num_bboxes, ).
    #         mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
    #             levels of a single image, each item has shape (num_bboxes, 4).
    #         mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
    #             each item has shape (num_bboxes, 4).
    #         level_ids (list[Tensor]): Indexes from all scale levels of a
    #             single image, each item has shape (num_bboxes, ).
    #         cfg (mmcv.Config): Test / postprocessing configuration,
    #             if None, `self.test_cfg` would be used.
    #         img_shape (tuple(int)): The shape of model's input image.
    #
    #     Returns:
    #         Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
    #             are bounding box positions (tl_x, tl_y, br_x, br_y) and the
    #             5-th column is a score between 0 and 1.
    #     """
    #     scores = torch.cat(mlvl_scores)
    #     anchors = torch.cat(mlvl_valid_anchors)
    #     rpn_bbox_pred = torch.cat(mlvl_bboxes)
    #     proposals = self.bbox_coder.decode(
    #         anchors, rpn_bbox_pred, max_shape=img_shape)
    #     ids = torch.cat(level_ids)
    #
    #     if cfg.min_bbox_size >= 0:
    #         w = proposals[:, 2] - proposals[:, 0]
    #         h = proposals[:, 3] - proposals[:, 1]
    #         valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
    #         if not valid_mask.all():
    #             proposals = proposals[valid_mask]
    #             scores = scores[valid_mask]
    #             ids = ids[valid_mask]
    #
    #     if proposals.numel() > 0:
    #         dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
    #     else:
    #         return proposals.new_zeros(0, 5)
    #
    #     return dets[:cfg.max_per_img]
