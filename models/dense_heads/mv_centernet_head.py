# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models import HEADS, CenterNetHead


@HEADS.register_module()
class MVCenterNetHead(CenterNetHead):

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (V.B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (V.B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (V.B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        views = len(img_metas[0]['scale_factor'])
        scale_factors = [scale_factor for img_meta in img_metas for scale_factor in img_meta['scale_factor']]

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        for img_id in range(len(img_metas)):
            for view in range(views):
                ori_shape = img_metas[img_id]['ori_shape'][view]
                h, w, _ = ori_shape
                if 'pad_to_centre' in img_metas[img_id].keys() and img_metas[img_id]['pad_to_centre']:
                    delta_x = max(0, h - w) // 2
                    delta_y = max(0, w - h) // 2
                    border = batch_det_bboxes.tensor([[delta_x, delta_y, delta_x, delta_y]]) \
                        .repeat_interleave(batch_det_bboxes[img_id * views + view].shape[0], dim=0)
                    batch_det_bboxes[img_id * views + view] -= border
        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (V.B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B.V, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (V.B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes for v_b_gt_bboxes in b_gt_bboxes]
        gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels for v_b_gt_labels in b_gt_labels]
        return super().loss(center_heatmap_preds, wh_preds, offset_preds, gt_bboxes, gt_labels, img_metas,
                            gt_bboxes_ignore)
