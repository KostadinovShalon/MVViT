import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.datasets.builder import build_dataset

import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect', 'MVFormatBundle', 'MVNormalize'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=999,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


view_colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    views = len(dataset[0]['filename'])
    for item in dataset:
        filenames = [os.path.join(args.output_dir, Path(filename).name) if args.output_dir is not None else None
                     for filename in item['filename']]
        for v in range(views):
            im = mmcv.imshow_det_bboxes(item['img'][v],
                                        item['gt_bboxes'][v],
                                        item['gt_labels'][v],
                                        class_names=dataset.CLASSES,
                                        show=not args.not_show,
                                        wait_time=args.show_interval,
                                        out_file=filenames[v],
                                        bbox_color=(255, 102, 61),
                                        text_color=(255, 102, 61))
            # if "fundamental_matrices" in item:
            #     h, w, _ = im.shape
            #     fundamental_matrices = item['fundamental_matrices'][v]
            #     for _v in range(views):
            #         if _v == v:
            #             continue
            #         f = fundamental_matrices[_v]
            #         v_gt_bboxes = item['gt_bboxes'][_v]
            #         v_gt_bboxes = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1] for bbox in v_gt_bboxes]
            #         epilines = [f @ np.array(b).transpose() for b in v_gt_bboxes]
            #         for epiline in epilines:
            #             cv2.line(im, (int(w/2), int(-(epiline[2] + epiline[0] * w / 2) / epiline[1])),
            #                      (w, int(-(epiline[2] + epiline[0] * w) / epiline[1])),
            #                      view_colours[_v])
            # cv2.imwrite(filenames[v], im)

        progress_bar.update()


if __name__ == '__main__':
    main()
