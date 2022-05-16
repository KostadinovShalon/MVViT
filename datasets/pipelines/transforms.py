import copy

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import find_inside_bboxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip, Pad, Normalize, RandomAffine, YOLOXHSVRandomAug, RandomCrop, \
    Mosaic

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class MVResize(Resize):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    @staticmethod
    def random_select(img_scales, seed=None):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        if seed is not None:
            np.random.seed(seed)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    def _random_scale(self, results, seed=None):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale, seed)
        else:
            raise NotImplementedError

        results['scale'] = tuple([scale] * len(results['img']))
        results['scale_idx'] = tuple([scale_idx] * len(results['img']))

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            imgs, w_scales, h_scales = [], [], []
            for img_member, scale in zip(results[key], results['scale']):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        img_member,
                        scale,
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = img_member.shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        img_member,
                        scale,
                        return_scale=True,
                        backend=self.backend)
                imgs.append(img)
                w_scales.append(w_scale)
                h_scales.append(h_scale)
            results[key] = tuple(imgs)
            img_shapes, scale_factors = [], []
            for img, w_scale, h_scale in zip(imgs, w_scales, h_scales):
                scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
                img_shapes.append(img.shape)
                scale_factors.append(scale_factor)
            results['img_shape'] = tuple(img_shapes)
            # in case that there is no padding
            results['pad_shape'] = tuple(img_shapes)
            results['scale_factor'] = tuple(scale_factors)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            mv_bboxes = []
            for bbox_member, scale_factor, img_shape \
                    in zip(results[key], results['scale_factor'], results['img_shape']):
                bboxes = bbox_member * scale_factor
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                mv_bboxes.append(bboxes)
            results[key] = tuple(mv_bboxes)

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            mv_masks = []
            for mask_member, img_shape, scale in zip(results[key], results['img_shape'], results['scale']):
                if self.keep_ratio:
                    mv_masks.append(mask_member.rescale(scale))
                else:
                    mv_masks.append(mask_member.resize(img_shape[:2]))
            results[key] = tuple(mv_masks)

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            mv_gt_segs = []
            for seg_member, scale in zip(results['seg_fields'], results['scale']):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        seg_member,
                        scale,
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        seg_member,
                        scale,
                        interpolation='nearest',
                        backend=self.backend)
                mv_gt_segs.append(gt_seg)
            results[key] = tuple(mv_gt_segs)

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                scales = []
                for img, scale_factor in zip(results['img'], results['scale_factor']):
                    img_shape = img.shape[:2]
                    assert isinstance(scale_factor, float)
                    scales.append(tuple([int(x * scale_factor) for x in img_shape][::-1]))
                results['scale'] = tuple(scales)
            else:
                seed = results['seed'] if 'seed' in results else None
                self._random_scale(results, seed)
        else:
            if not self.override:
                assert 'scale_factors' not in results, (
                    'scales and scale_factors cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                seed = results['seed'] if 'seed' in results else None
                self._random_scale(results, seed)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class MVRandomCrop(RandomCrop):

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0

        for key in results.get('img_fields', ['img']):
            imgs = []
            img_shapes = []
            for img in results[key]:
                margin_h = max(img.shape[0] - crop_size[0], 0)
                margin_w = max(img.shape[1] - crop_size[1], 0)
                offset_h = np.random.randint(0, margin_h + 1)
                offset_w = np.random.randint(0, margin_w + 1)
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

                # crop the image
                img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
                img_shape = img.shape
                imgs.append(img)
                img_shapes.append(img_shape)
            results[key] = tuple(imgs)
        results['img_shape'] = tuple(img_shapes)

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)

            bbox_field = []
            valid_inds = []
            for bbox, img_shape in zip(results[key], img_shapes):
                bboxes = bbox - bbox_offset
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                valid_ind = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1])
                # If the crop does not contain any gt-bbox area and
                # allow_negative_crop is False, skip this image.
                if (key == 'gt_bboxes' and not valid_ind.any()
                        and not allow_negative_crop):
                    return None
                bbox_field.append(bboxes[valid_ind, :])
                valid_inds.append(valid_ind)
            results[key] = tuple(bbox_field)
            # label fields. e.g. gt_labels and gt_labels_ignore

            label_key = self.bbox2label.get(key)
            if label_key in results:
                label_keys = []
                for r, valid_ind in zip(results[label_key], valid_inds):
                    label_keys.append(r[valid_ind])
                results[label_key] = tuple(label_keys)

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                raise NotImplementedError("Mask not implemented")
                # results_masks = []
                # for r, valid_ind in zip(results[mask_key], valid_inds):
                #     results_masks.append(r[
                #         valid_ind.nonzero()[0]].crop(
                #             np.asarray([crop_x1, crop_y1, crop_x2, crop_y2])))
                # results[mask_key] = tuple(results_masks)
                # if self.recompute_bbox:
                #     results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            raise NotImplementedError("Segmentation not implemented")
            # results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'][0].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results


@PIPELINES.register_module()
class MVRandomFlip(RandomFlip):
    """Flip the images & bboxes & masks.
    """

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        cur_dir = None
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                flipped_imgs = []
                for img in results[key]:
                    flipped_imgs.append(mmcv.imflip(img, direction=results['flip_direction']))
                results[key] = tuple(flipped_imgs)
            # flip bboxes
            for key in results.get('bbox_fields', []):
                flipped_bboxes = []
                for bbox_member, img_shape in zip(results[key], results['img_shape']):
                    flipped_bboxes.append(self.bbox_flip(bbox_member,
                                                         img_shape,
                                                         results['flip_direction']))
                results[key] = tuple(flipped_bboxes)
            # flip masks
            for key in results.get('mask_fields', []):
                flipped_masks = []
                for mask in results[key]:
                    flipped_masks.append(mask.flip(results['flip_direction']))
                results[key] = tuple(flipped_masks)

            # flip segs
            for key in results.get('seg_fields', []):
                flipped_segs = []
                for seg in results[key]:
                    flipped_segs.append(mmcv.imflip(seg, direction=results['flip_direction']))
                results[key] = tuple(flipped_segs)
        return results


@PIPELINES.register_module()
class MVPad(Pad):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shapes", "pad_fixed_size", "pad_size_divisor",

    PADDING IS ALWAYS DONE ON RIGHT AND BOTTOM, SO BBOXES ARE NOT MODIFIED
    """

    def __init__(self, size=None, size_divisor=None, pad_val=dict(img=0, masks=0, seg=255), pad_to_centre=False, pad_to_square=False):
        super().__init__(size, size_divisor, pad_to_square, pad_val)
        self.pad_to_centre = pad_to_centre

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            padded_imgs = []
            for img in results[key]:
                if self.pad_to_square:
                    max_size = max(img.shape[:2])
                    self.size = (max_size, max_size)
                if self.size is not None:
                    if self.pad_to_centre:
                        h, w, _ = img.shape
                        size_w, size_h = self.size if isinstance(self.size, tuple) else (self.size, self.size)
                        delta_w = max(0, size_w - w)
                        delta_h = max(0, size_h - h)
                        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
                        padded_imgs.append(mmcv.impad(img, padding=padding, pad_val=pad_val))
                    else:
                        padded_imgs.append(mmcv.impad(img, shape=self.size, pad_val=pad_val))
                elif self.size_divisor is not None:
                    padded_imgs.append(mmcv.impad_to_multiple(img, self.size_divisor, pad_val=pad_val))
            results[key] = tuple(padded_imgs)
        results['pad_shape'] = tuple(padded_img.shape for padded_img in results['img'])

        if self.pad_to_centre and self.size is not None:
            # Pad bboxes
            for key in results.get('bbox_fields', []):
                transformed_bboxes = []
                for bbox_member, img_shape, pad_shape in zip(results[key], results['img_shape'], results['pad_shape']):
                    h, w, _ = img_shape
                    pad_h, pad_w, _ = pad_shape
                    delta_w = max(0, pad_w - w) // 2
                    delta_h = max(0, pad_h - h) // 2
                    delta = np.array([[delta_w, delta_h, delta_w, delta_h]], dtype=np.float32)\
                        .repeat(len(bbox_member), axis=0)
                    transformed_bboxes.append(bbox_member + delta)
                results[key] = tuple(transformed_bboxes)

        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['pad_to_centre'] = self.pad_to_centre

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = self.pad_val.get('masks', 0)
        pad_shapes = [pad_shape[:2] for pad_shape in results['pad_shape']]
        for key in results.get('mask_fields', []):
            padded_masks = []
            for mask, pad_shape in zip(results[key], pad_shapes):
                padded_masks.append(mask.pad(pad_shape, pad_val=pad_val))
            results[key] = tuple(padded_masks)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_shapes = [pad_shape[:2] for pad_shape in results['pad_shape']]
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            padded_segs = []
            for seg, pad_shape in zip(results[key], pad_shapes):
                padded_segs.append(mmcv.impad(seg, shape=pad_shape[:2], pad_val=pad_val))
            results[key] = tuple(padded_segs)


@PIPELINES.register_module()
class MVNormalize(Normalize):
    """Normalize the images."""

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            imgs = []
            for img in results[key]:
                imgs.append(mmcv.imnormalize(img, self.mean, self.std, self.to_rgb))
            results[key] = tuple(imgs)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class MVRandomAffine(RandomAffine):

    def __call__(self, results):
        imgs = []
        img_shapes = []
        bboxes = {key: list() for key in results.get('bbox_fields', [])}
        gt_labels = []
        for img in results['img']:
            height = img.shape[0] + self.border[0] * 2
            width = img.shape[1] + self.border[1] * 2

            # Rotation
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                             self.max_rotate_degree)
            rotation_matrix = self._get_rotation_matrix(rotation_degree)

            # Scaling
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                           self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)

            # Shear
            x_degree = random.uniform(-self.max_shear_degree,
                                      self.max_shear_degree)
            y_degree = random.uniform(-self.max_shear_degree,
                                      self.max_shear_degree)
            shear_matrix = self._get_shear_matrix(x_degree, y_degree)

            # Translation
            trans_x = random.uniform(-self.max_translate_ratio,
                                     self.max_translate_ratio) * width
            trans_y = random.uniform(-self.max_translate_ratio,
                                     self.max_translate_ratio) * height
            translate_matrix = self._get_translation_matrix(trans_x, trans_y)

            warp_matrix = (
                translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

            img = cv2.warpPerspective(
                img,
                warp_matrix,
                dsize=(width, height),
                borderValue=self.border_val)
            imgs.append(img)
            img_shapes.append(img.shape)

            for key in results.get('bbox_fields', []):
                for i, bbox in enumerate(results[key]):
                    # bboxes = results[key]
                    num_bboxes = len(bbox)
                    if num_bboxes:
                        # homogeneous coordinates
                        xs = bbox[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                        ys = bbox[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                        ones = np.ones_like(xs)
                        points = np.vstack([xs, ys, ones])

                        warp_points = warp_matrix @ points
                        warp_points = warp_points[:2] / warp_points[2]
                        xs = warp_points[0].reshape(num_bboxes, 4)
                        ys = warp_points[1].reshape(num_bboxes, 4)

                        warp_bboxes = np.vstack(
                            (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                        if self.bbox_clip_border:
                            warp_bboxes[:, [0, 2]] = \
                                warp_bboxes[:, [0, 2]].clip(0, width)
                            warp_bboxes[:, [1, 3]] = \
                                warp_bboxes[:, [1, 3]].clip(0, height)

                        # remove outside bbox
                        valid_index = find_inside_bboxes(warp_bboxes, height, width)
                        if not self.skip_filter:
                            # filter bboxes
                            filter_index = self.filter_gt_bboxes(
                                bbox * scaling_ratio, warp_bboxes)
                            valid_index = valid_index & filter_index

                        bboxes[key].append(warp_bboxes[valid_index])
                        if key in ['gt_bboxes']:
                            if 'gt_labels' in results:
                                gt_labels.append(results['gt_labels'][i][valid_index])

                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomAffine only supports bbox.')

        results['img'] = tuple(imgs)
        results['img_shape'] = tuple(img_shapes)
        if len(gt_labels) > 0:
            results['gt_labels'] = tuple(gt_labels)
        for key in results.get('bbox_fields', []):
            results[key] = tuple(bboxes[key])
        return results


@PIPELINES.register_module()
class MVYOLOXHSVRandomAug(YOLOXHSVRandomAug):
    def __call__(self, results):
        imgs = []
        for img in results['img']:
            hsv_gains = np.random.uniform(-1, 1, 3) * [
                self.hue_delta, self.saturation_delta, self.value_delta
            ]
            # random selection of h, s, v
            hsv_gains *= np.random.randint(0, 2, 3)
            # prevent overflow
            hsv_gains = hsv_gains.astype(np.int16)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
            cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
            imgs.append(img)
        results['img'] = tuple(imgs)
        return results


@PIPELINES.register_module()
class MVMosaic(Mosaic):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
    """

    # def __init__(self,
    #              img_scale=(640, 640),
    #              center_ratio_range=(0.5, 1.5),
    #              min_bbox_size=0,
    #              bbox_clip_border=True,
    #              skip_filter=True,
    #              pad_val=114):

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        views = len(results['img'])
        mosaic_imgs = []
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')

        for v in range(views):
            v_mosaic_labels = []
            v_mosaic_bboxes = []
            if len(results['img'][v].shape) == 3:
                mosaic_img = np.full(
                    (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                    self.pad_val,
                    dtype=results['img'][v].dtype)
            else:
                mosaic_img = np.full(
                    (int(self.img_scale[v] * 2), int(self.img_scale[1] * 2)),
                    self.pad_val,
                    dtype=results['img'][v].dtype)

            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    results_patch = copy.deepcopy(results)
                else:
                    results_patch = copy.deepcopy(results['mix_results'][i - 1])

                img_i = results_patch['img'][v]
                h_i, w_i = img_i.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                img_i = mmcv.imresize(
                    img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, img_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

                # adjust coordinate
                gt_bboxes_i = results_patch['gt_bboxes'][v]
                gt_labels_i = results_patch['gt_labels'][v]

                if gt_bboxes_i.shape[0] > 0:
                    padw = x1_p - x1_c
                    padh = y1_p - y1_c
                    gt_bboxes_i[:, 0::2] = \
                        scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                    gt_bboxes_i[:, 1::2] = \
                        scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

                v_mosaic_bboxes.append(gt_bboxes_i)
                v_mosaic_labels.append(gt_labels_i)

            if len(v_mosaic_labels) > 0:
                v_mosaic_bboxes = np.concatenate(v_mosaic_bboxes, 0)
                v_mosaic_labels = np.concatenate(v_mosaic_labels, 0)

                if self.bbox_clip_border:
                    v_mosaic_bboxes[:, 0::2] = np.clip(v_mosaic_bboxes[:, 0::2], 0,
                                                     2 * self.img_scale[1])
                    v_mosaic_bboxes[:, 1::2] = np.clip(v_mosaic_bboxes[:, 1::2], 0,
                                                     2 * self.img_scale[0])

                if not self.skip_filter:
                    v_mosaic_bboxes, v_mosaic_labels = \
                        self._filter_box_candidates(v_mosaic_bboxes, v_mosaic_labels)

            # remove outside bboxes
            inside_inds = find_inside_bboxes(v_mosaic_bboxes, 2 * self.img_scale[0],
                                             2 * self.img_scale[1])
            v_mosaic_bboxes = v_mosaic_bboxes[inside_inds]
            v_mosaic_labels = v_mosaic_labels[inside_inds]

            mosaic_imgs.append(mosaic_img)
            mosaic_labels.append(v_mosaic_labels)
            mosaic_bboxes.append(v_mosaic_bboxes)

        results['img'] = tuple(mosaic_imgs)
        results['img_shape'] = tuple(mosaic_img.shape for mosaic_img in mosaic_imgs)
        results['gt_bboxes'] = tuple(mosaic_bboxes)
        results['gt_labels'] = tuple(mosaic_labels)

        return results
