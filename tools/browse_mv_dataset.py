import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a multi-view dataset')
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
            mmcv.imshow_det_bboxes(item['img'][v],
                                   item['gt_bboxes'][v],
                                   item['gt_labels'][v],
                                   class_names=dataset.CLASSES,
                                   show=not args.not_show,
                                   wait_time=args.show_interval,
                                   out_file=filenames[v],
                                   bbox_color=(255, 102, 61),
                                   text_color=(255, 102, 61))
        progress_bar.update()


if __name__ == '__main__':
    main()
