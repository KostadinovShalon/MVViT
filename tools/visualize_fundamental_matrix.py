import argparse
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt, gridspec
from mmcv import Config

from mmdet.datasets.builder import build_dataset


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
        '--test-idx',
        type=int,
        default=-1,
        help='Index to test. If not given, a random index will be chosen.')
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
    idx = args.test_idx
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)
    if idx == -1:
        idx = random.randint(0, len(dataset))

    item = dataset[idx]
    fundamental_matrixes = item['fundamental_matrices']

    for from_view, fs in enumerate(fundamental_matrixes):
        for to_view, F in fs.items():
            from_bboxes = item['gt_bboxes'][from_view]  # list of lists
            from_centres = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1] for bbox in from_bboxes]
            from_centres = np.array(from_centres)  # n x 2 matrix

            to_bboxes = item['gt_bboxes'][to_view]
            to_centres = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1] for bbox in to_bboxes]
            to_centres = np.array(to_centres)  # n x 3 matrix

            epipolar_lines = np.array(F) @ from_centres.transpose()

            fig = plt.figure()
            gs = gridspec.GridSpec(1, 2)

            from_img = np.copy(item['img'][from_view])
            to_img = np.copy(item['img'][to_view])

            for bbox in from_bboxes:
                cv2.rectangle(from_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
            for bbox in to_bboxes:
                cv2.rectangle(to_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
            for epipolar_line in epipolar_lines.transpose():
                A, B, C = epipolar_line
                ymax = item['pad_shape'][to_view][0] - 1
                p0 = (int(-C / A), 0)
                p1 = (int(-(B*ymax + C) / A), ymax)
                cv2.line(to_img, p0, p1, (0, 0, 255), thickness=2)

            ax = fig.add_subplot(gs[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(from_img)

            ax = fig.add_subplot(gs[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(to_img)

            plt.show()



if __name__ == '__main__':
    main()
