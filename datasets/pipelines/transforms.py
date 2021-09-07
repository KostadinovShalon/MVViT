import mmcv
import numpy as np
from numpy import random

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip, Pad, Normalize

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

    def __init__(self, size=None, size_divisor=None, pad_val=0, pad_to_centre=False, pad_to_square=False):
        super().__init__(size, size_divisor, pad_to_square, pad_val)
        self.pad_to_centre = pad_to_centre

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
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
                        padded_imgs.append(mmcv.impad(img, padding=padding, pad_val=self.pad_val))
                    else:
                        padded_imgs.append(mmcv.impad(img, shape=self.size, pad_val=self.pad_val))
                elif self.size_divisor is not None:
                    padded_imgs.append(mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val))
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
        pad_shapes = [pad_shape[:2] for pad_shape in results['pad_shape']]
        for key in results.get('mask_fields', []):
            padded_masks = []
            for mask, pad_shape in zip(results[key], pad_shapes):
                padded_masks.append(mask.pad(pad_shape, pad_val=self.pad_val))
            results[key] = tuple(padded_masks)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_shapes = [pad_shape[:2] for pad_shape in results['pad_shape']]
        for key in results.get('seg_fields', []):
            padded_segs = []
            for seg, pad_shape in zip(results[key], pad_shapes):
                padded_segs.append(mmcv.impad(seg, shape=pad_shape[:2]))
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
