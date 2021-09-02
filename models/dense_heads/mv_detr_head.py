import torch

import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import reduce_mean
from mmdet.models import HEADS, DETRHead


@HEADS.register_module()
class MVDETRHead(DETRHead):

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, v, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs.v, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs.v, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        views = x.size(1)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, views, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for view in range(views):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][view]
                masks[img_id, view, :img_h, :img_w] = 0

        x = x.transpose(0, 1).contiguous()  # V B
        x = torch.stack([self.input_proj(xv) for xv in x])
        x = x.transpose(0, 1).contiguous()  # B V
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=(x.shape[1], x.shape[-2], x.shape[-1])).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, v, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, views, num_query, embed_dim]  TODO: continuar aqui
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)
        n, bs, v, nq, d = outs_dec.shape
        outs_dec = outs_dec.view(n, -1, nq, d)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gt_bboxes = [v_b_gt_bboxes for b_gt_bboxes in gt_bboxes_list for v_b_gt_bboxes in b_gt_bboxes]
        gt_labels = [v_b_gt_labels for b_gt_labels in gt_labels_list for v_b_gt_labels in b_gt_labels]
        img_metas = [{k: (val[v] if isinstance(val, list) else val) for k, val in img_meta.items()} for img_meta in img_metas for v in range(len(img_meta['img_shape']))]
        return super().loss(all_cls_scores_list,
                            all_bbox_preds_list,
                            gt_bboxes,
                            gt_labels,
                            img_metas,
                            gt_bboxes_ignore)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        views = len(img_metas[0]['img_shape'])

        for img_id in range(len(img_metas)):
            for v in range(views):
                cls_score = cls_scores[img_id * views + v]
                bbox_pred = bbox_preds[img_id * views + v]
                img_shape = img_metas[img_id]['img_shape'][v]
                scale_factor = img_metas[img_id]['scale_factor'][v]
                proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                    img_shape, scale_factor,
                                                    rescale)
                result_list.append(proposals)

        return result_list
