import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import print_log

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class CustomMVDataset(CustomDataset):
    """Custom multi-view dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': ('a.jpg', 'b.jpg'),
                'width': (1280, 1280),
                'height': (720, 720),
                'ann': ({
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }, ...)
            },
            ...
        ]

    Args:
        ann_file (tuple[str]): List of annotation file paths.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    def __init__(self,
                 ann_files,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_files=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 skip_type_keys=None):
        self.ann_files = ann_files  # list
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_files = proposal_files  # list
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)
        self.views = len(self.ann_files)

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        # join paths if data_root is specified
        if self.data_root is not None:
            self.ann_files = tuple(osp.join(self.data_root, ann_file) if not osp.isabs(ann_file) else ann_file
                                   for ann_file in self.ann_files)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if self.proposal_file is not None:
                self.proposal_file = tuple(
                    osp.join(self.data_root, proposal_file) if not osp.isabs(proposal_file) else proposal_file
                    for proposal_file in self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            self.data_infos = []
            for ann_file in self.ann_files:
                with self.file_client.get_local_path(ann_file) as local_path:
                    self.data_infos.append(self.load_annotations(local_path))
            self.data_infos = tuple(self.data_infos)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_files} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = tuple(self.load_annotations(ann_file) for ann_file in self.ann_files)  # Tuple of data infos
        size = len(self.data_infos[0])
        for i in range(1, self.views):
            assert len(self.data_infos[i]) == size

        if self.proposal_files is not None:
            if hasattr(self.file_client, 'get_local_path'):
                self.proposals = []
                for proposal_file in self.proposal_files:
                    with self.file_client.get_local_path(proposal_file) as local_path:
                        self.proposals.append(self.load_proposals(local_path))
                self.proposals = tuple(self.proposals)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.proposal_files} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = tuple(
                    self.load_proposals(proposal_file) for proposal_file in self.proposal_files)  # Tuple of data infos
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = tuple([self.data_infos[v][i] for i in valid_inds] for v in range(len(self.data_infos)))
            if self.proposals is not None:
                self.proposals = tuple([self.proposals[v][i] for i in valid_inds] for v in range(len(self.proposals)))
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)


    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_files
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos[0])

    def get_anns_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            tuple[dict]: A tuple of dicts wit annotation info of specified index on each view.
        """

        return tuple(data_info[idx]['ann'] for data_info in self.data_infos)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            tuple[list[int]]: All categories in the image of specified index.
        """

        return tuple(data_info[idx]['ann']['labels'].astype(np.int).tolist() for data_info in self.data_infos)

    def _filter_imgs(self, min_size=32):
        """Filter images too small. Returns a list of lists"""
        if self.filter_empty_gt:
            warnings.warn(
                'MVCustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for data_info in self.data_infos:
            for i, img_info in enumerate(data_info):
                if min(img_info['width'], img_info['height']) >= min_size:
                    valid_inds.append(i)
        return list(set(valid_inds))

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[0][i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx, view=0):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag[:, view] == self.flag[idx, view])[0]
        return np.random.choice(pool)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            tuple[dict]: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        imgs_info = tuple(data_info[idx] for data_info in self.data_infos)
        anns_info = self.get_anns_info(idx)
        results = dict(img_info=imgs_info, ann_info=anns_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        imgs_info = tuple(data_info[idx] for data_info in self.data_infos)
        results = dict(img_info=imgs_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thr, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
