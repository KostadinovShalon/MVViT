import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config
import numpy as np

from mmdet.datasets.builder import build_dataset
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['MVFormatBundle', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    views = len(dataset[0]['filename'])
    pts = []
    filenames = []
    F = {(v, _v): None for v in range(views - 1) for _v in range(v + 1, views)}
    for item in dataset:
        filenames.append([os.path.join(args.output_dir, Path(filename).name) if args.output_dir is not None else None
                     for filename in item['filename']])
        item_pts = []
        for gt_bboxes in item['gt_bboxes']:
            gt = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in gt_bboxes]  # list of x, y
            item_pts.append(gt)
        item_pts = list(map(list, zip(*item_pts)))
        pts += item_pts
        progress_bar.update()
    for v in range(views - 1):
        for _v in range(v + 1, views):
            src_view_points = np.array([pt[v] for pt in pts])
            dst_view_points = np.array([pt[_v] for pt in pts])
            Ft, _ = ransac((src_view_points, dst_view_points), FundamentalMatrixTransform, min_samples=8,
                           residual_threshold=1, max_trials=5000)
            F[(v, _v)] = Ft.params
    print(F)


if __name__ == '__main__':
    main()
