# Copyright (c) 2019 Western Digital Corporation or its affiliates.
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolo_head import YOLOV3Head

import torch


@HEADS.register_module()
class YOLOV3MVHead(YOLOV3Head):

    @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        views = len(pred_maps[0])
        for v in range(views):
            pred_maps_list = [
                pred_maps[i][v::views].detach() for i in range(num_levels)
            ]
            scale_factors = [
                img_metas[i]['scale_factor'][v]
                for i in range(pred_maps_list[0].shape[0])
            ]
            proposals = super().get_bboxes(pred_maps_list, scale_factors,
                                         cfg, rescale, with_nms)
            for img_id in range(len(img_metas)):
                if rescale and 'pad_to_centre' in img_metas[img_id].keys() and img_metas[img_id]['pad_to_centre']:
                    ori_shape = img_metas[img_id]['ori_shape'][v]
                    h, w, _ = ori_shape
                    delta_x = max(0, h - w) // 2
                    delta_y = max(0, w - h) // 2
                    remove_padding = torch.tensor([[-delta_x, -delta_y, -delta_x, -delta_y, 0]]) \
                        .repeat_interleave(proposals[img_id][0].shape[0], dim=0).to(proposals[img_id][0].device)
                    proposals[img_id] = proposals[img_id][0] + remove_padding, proposals[img_id][1]
                result_list.append(proposals[img_id])
        return result_list

    @force_fp32(apply_to=('pred_maps',))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes for v_b_gt_bboxes in b_gt_bboxes]
        gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels for v_b_gt_labels in b_gt_labels]
        num_imgs = len(gt_bboxes)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(num_imgs):
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)
