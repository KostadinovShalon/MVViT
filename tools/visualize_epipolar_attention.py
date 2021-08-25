import argparse

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmdet.models import build_detector

import random
import os

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint')
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
    parser.add_argument(
        '--idx',
        type=int,
        default=-1, )
    parser.add_argument(
        '--from_view',
        type=int,
        default=0, )
    parser.add_argument(
        '--to_view',
        type=int,
        default=1, )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    idx = args.idx
    if idx == -1:
        idx = random.randint(0, len(data_loader))
    from_view = args.from_view
    to_view = args.to_view
    for _idx, x in enumerate(data_loader):
        if _idx != idx:
            continue
        x = x['img'].data[0].to(torch.device('cuda:0'))
        x_ft = x.clone().detach()

        with torch.no_grad():
            mvdarknet = model.module.backbone
            sims = mvdarknet.get_sims(x_ft, from_view=from_view, to_view=to_view)

            ref_view = x[0, from_view, ...].permute(1, 2, 0).cpu().numpy()
            w, h = x.shape[3], x.shape[4]
            gw = w // mvdarknet.fw
            gh = h // mvdarknet.fh

            for i, pts in enumerate(tqdm.tqdm(mvdarknet.sampling_points[(from_view, to_view)])):
                yi = i % mvdarknet.fw
                xi = i // mvdarknet.fh

                if yi < mvdarknet.fw / 4 or yi > 3 * mvdarknet.fw / 4:
                    continue
                if xi < mvdarknet.fh / 4:
                    continue

                _ref_view = cv2.rectangle(ref_view.copy(), (xi * gw, yi * gh), ((xi + 1) * gw, (yi + 1) * gh),
                                          (1, 0, 0))
                # _ref_view = cv2.cvtColor(_ref_view, cv2.COLOR_RGB2BGR)

                heads = len(sims) if isinstance(sims, list) else 1

                fig = plt.figure(figsize=(10 * (heads + 1), 10))
                gs = gridspec.GridSpec(2, heads + 1)
                gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

                ax = fig.add_subplot(gs[0])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(_ref_view)

                _src_view = x[0, to_view, ...].permute(1, 2, 0).cpu().numpy().copy()

                x_line = [int(np.round((pt[0].item() + 1) * w / 2)) for pt in pts]
                y_line = [int(np.round((pt[1].item() + 1) * h / 2)) for pt in pts]

                m = (y_line[-1] - y_line[0]) / (x_line[-1] - x_line[0] + 1e-16)

                if not isinstance(sims, list):
                    sims = [sims]

                for h_idx, sim in enumerate(sims):
                    output = cv2.line(_src_view.copy(), (x_line[0], y_line[0]), (x_line[-1], y_line[-1]), 0, 1)
                    # scale = 100
                    # s = scale * sim[0, xi, yi].squeeze(0).cpu().numpy() + 5
                    # xx = np.array(x_line) + abs(m) * s / np.sqrt(m ** 2 + 1)
                    # yy = np.array(y_line) - s / np.sqrt(m ** 2 + 1)
                    # for k, (xl, yl) in enumerate(zip(x_line, y_line)):
                    #     s = sim[0, xi, yi, k].cpu().item()
                    #     overlay = cv2.circle(output.copy(), (xl, yl), gw // 2, (1, 0, 0), -1)
                    #     cv2.addWeighted(overlay, s, output, 1 - s, 0, output)
                    ax = fig.add_subplot(gs[h_idx + 1])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(output)
                    ax2 = fig.add_subplot(gs[heads + 1 + h_idx + 1])
                    ax2.plot(sim[0, xi, yi].squeeze(0).cpu().numpy())
                    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
                    ax2.set_aspect(asp)

                os.makedirs(os.path.join(args.output_dir, f"test{idx}"), exist_ok=True)
                plt.savefig(os.path.join(args.output_dir, f"test{idx}", f"{i}.jpg"), bbox_inches='tight')
                plt.close(fig)
        break


if __name__ == '__main__':
    main()
