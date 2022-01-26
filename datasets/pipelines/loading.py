import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations


@PIPELINES.register_module()
class LoadMVImagesFromFile(LoadImageFromFile):
    """Load a tuple of same-scene-different-view images from files. The difference from mmdet LoadImageFromFile is
    that dict values are converted to tuples containing the information from all the views.

    """
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filenames = tuple(osp.join(results['img_prefix'], r['filename']) for r in results['img_info'])
        else:
            filenames = tuple(r['filename'] for r in results['img_info'])

        imgs_bytes = tuple(self.file_client.get(filename) for filename in filenames)
        imgs = tuple(mmcv.imfrombytes(img_bytes, flag=self.color_type) for img_bytes in imgs_bytes)
        if self.to_float32:
            imgs = tuple(img.astype(np.float32) for img in imgs)

        results['filename'] = filenames
        results['ori_filename'] = tuple(r['filename'] for r in results['img_info'])
        results['img'] = imgs
        results['img_shape'] = tuple(img.shape for img in imgs)
        results['ori_shape'] = tuple(img.shape for img in imgs)
        results['img_field'] = ['img']
        results['seed'] = np.random.randint(10000)
        return results


@PIPELINES.register_module()
class LoadMVAnnotations(LoadAnnotations):
    """Load multi-view annotations.
    """

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations. Currently bboxes_ignore is not implemented

        Args:
            results (dict): Result dict from :obj:`CustomMVDataset`.

        Returns:
            dict: The dict contains loaded multi-view bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = tuple(sv_ann_info['bboxes'].copy() for sv_ann_info in ann_info)

        # TODO: Implement bbox ignore
        # gt_bboxes_ignore = tuple(sv_ann_info.get('bboxes_ignore', None) for sv_ann_info in ann_info)
        # for i, ignore in enumerate(gt_bboxes_ignore):
        #     if ignore is not None:
        #         results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
        #         results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load multi-view label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = tuple(sv_ann_info['labels'].copy() for sv_ann_info in results['ann_info'])
        return results

    def _load_masks(self, results):
        raise NotImplementedError()
        # """Private function to load mask annotations.
        #
        # Args:
        #     results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        #
        # Returns:
        #     dict: The dict contains loaded mask annotations.
        #         If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
        #         :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        # """
        #
        # h, w = results['img_info']['height'], results['img_info']['width']
        # gt_masks = results['ann_info']['masks']
        # if self.poly2mask:
        #     gt_masks = BitmapMasks(
        #         [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        # else:
        #     gt_masks = PolygonMasks(
        #         [self.process_polygons(polygons) for polygons in gt_masks], h,
        #         w)
        # results['gt_masks'] = gt_masks
        # results['mask_fields'].append('gt_masks')
        # return results

    def _load_semantic_seg(self, results):
        raise NotImplementedError()
        # """Private function to load semantic segmentation annotations.
        #
        # Args:
        #     results (dict): Result dict from :obj:`dataset`.
        #
        # Returns:
        #     dict: The dict contains loaded semantic segmentation annotations.
        # """
        #
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)
        #
        # filename = osp.join(results['seg_prefix'],
        #                     results['ann_info']['seg_map'])
        # img_bytes = self.file_client.get(filename)
        # results['gt_semantic_seg'] = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged').squeeze()
        # results['seg_fields'].append('gt_semantic_seg')
        # return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        assert not self.with_mask, "with_mask option not implemented yet"
        assert not self.with_seg, "with_seg option not implemented yet"
        return super().__call__(results)

