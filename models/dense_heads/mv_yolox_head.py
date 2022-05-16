# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.runner import force_fp32

from mmdet.core import (multi_apply,
                        reduce_mean)
from mmdet.models import YOLOXHead, HEADS


@HEADS.register_module()
class MVYOLOXHead(YOLOXHead):

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
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
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        num_levels = len(bbox_preds)
        views = len(bbox_preds[0])

        result_list = []
        for v in range(views):
            cls_scores_list = [
                cls_scores[i][v::views].detach() for i in range(num_levels)
            ]
            bbox_preds_list = [
                bbox_preds[i][v::views].detach() for i in range(num_levels)
            ]
            objectnesses_list = [
                objectnesses[i][v::views].detach() for i in range(num_levels)
            ]
            scale_factors = [
                img_metas[i]['scale_factor'][v]
                for i in range(bbox_preds_list[0].shape[0])
            ]

            num_imgs = len(img_metas)
            featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores_list]
            mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores_list[0].dtype,
                device=cls_scores_list[0].device,
                with_stride=True)

            # flatten cls_scores, bbox_preds and objectness
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                      self.cls_out_channels)
                for cls_score in cls_scores_list
            ]
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                for bbox_pred in bbox_preds_list
            ]
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses_list
            ]

            flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
            flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
            flatten_priors = torch.cat(mlvl_priors)

            flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
            if rescale:
                flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
                    scale_factors).unsqueeze(1)

            for img_id in range(len(img_metas)):
                im_cls_scores = flatten_cls_scores[img_id]
                im_score_factor = flatten_objectness[img_id]
                im_bboxes = flatten_bboxes[img_id]
                result_list.append(
                    self._bboxes_nms(im_cls_scores, im_bboxes, im_score_factor, cfg))

        return result_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes for v_b_gt_bboxes in b_gt_bboxes]
        gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels for v_b_gt_labels in b_gt_labels]
        num_imgs = len(gt_bboxes)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict
