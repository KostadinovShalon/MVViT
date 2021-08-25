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
    parser = argparse.ArgumentParser(description='Tests the Fundamental Matrix')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['MVFormatBundle', 'Collect'],
        help='skip some useless pipeline')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def distance_point_line(p, l):
    """
    p -> 3d vector
    l -> 3d vector
    """
    return np.abs(np.dot(p, l)) / np.linalg.norm(l[:-1])


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    fundamental_matrices = dataset.fundamental_matrices
    f_errors = {views: list() for views in fundamental_matrices.keys()}
    for item in dataset:
        for views, f in fundamental_matrices.items():
            from_view, to_view = views

            from_bboxes = item['gt_bboxes'][from_view]  # list of lists
            from_centres = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1] for bbox in from_bboxes]
            from_centres = np.array(from_centres)  # n x 3 matrix

            to_bboxes = item['gt_bboxes'][to_view]
            to_centres = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1] for bbox in to_bboxes]
            to_centres = np.array(to_centres)  # n x 3 matrix

            for from_centre, to_centre in zip(from_centres, to_centres):
                error = distance_point_line(from_centre, to_centre @ f) ** 2
                error += distance_point_line(to_centre, from_centre @ np.transpose(f)) ** 2
                f_errors[views].append(error)

        progress_bar.update()

    for keys, errors in f_errors.items():
        error = np.sum(errors) / len(errors)
        print(f"Symmetric residual error for views {keys}: {error}")


if __name__ == '__main__':
    main()
