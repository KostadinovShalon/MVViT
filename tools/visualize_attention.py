import os

import mmcv
import torch
import tqdm
from mmcv import Config
from mmcv.ops import RoIPool
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from scipy.interpolate import griddata

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt, gridspec
import scipy


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
        '--x',
        default=None,
        type=int,)
    parser.add_argument(
        '--y',
        default=None,
        type=int,)
    parser.add_argument(
        '--views_paths',
        type=str,
        default=None,
        nargs='+')
    args = parser.parse_args()
    return args


def draw_attention(ref_view, src_view, attention_weights, out_dir, x_pos=None, y_pos=None):
    # views: C x W x H
    # attention_weights: H x W x H x W
    h, w = ref_view.shape[1], ref_view.shape[2]
    sh, sw = src_view.shape[1], src_view.shape[2]
    ref_view = ref_view.clone().permute(1, 2, 0).cpu().numpy()
    src_view = src_view.clone().permute(1, 2, 0).cpu().numpy()


    # if fusion == "mean":
    #     attention_heads_fused = attention_weights.mean(axis=0)
    # elif fusion == "max":
    #     attention_heads_fused = attention_weights.max(axis=0)[0]
    # elif fusion == "min":
    #     attention_heads_fused = attention_weights.min(axis=0)[0]
    attention_weights = attention_weights.clone().cpu()
    attention_weights -= attention_weights.min()
    attention_weights /= attention_weights.max()

    for yi in tqdm.tqdm(range(attention_weights.size(0))):
        for xi in tqdm.tqdm(range(attention_weights.size(1)), leave=False):

            if x_pos is not None and y_pos is not None:
                yi = y_pos
                xi = x_pos

            attn = attention_weights[yi, xi]
            grid_size = attn.shape
            gw = w // grid_size[1]
            gh = h // grid_size[0]

            _ref_view = cv2.rectangle(ref_view.copy(), (xi * gw, yi * gh), ((xi + 1) * gw, (yi + 1) * gh),
                                      (1, 0, 0))

            fig = plt.figure(figsize=(20, 40))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

            ax = fig.add_subplot(gs[0, 0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(_ref_view)


            # attn = attn.view(-1)
            #
            # _, min_indices = attn.topk(
            #     int(attn.size(-1) * discard_ratio), -1, False
            # )
            # attn = attn.scatter(0, min_indices, 0)
            # attn = attn.view(grid_size)

            attn_plot = np.zeros((sw, sh), np.float32)
            # avg_attention_weights = attention_weights.sum(dim=0) / heads
            # avg_attention_weights = avg_attention_weights[i] # - avg_attention_weights[i].min()
            # avg_attention_weights /= avg_attention_weights.max()
            grid_x, grid_y = np.mgrid[0:w, 0:h]
            points_x, points_y = np.mgrid[(gw//2):w:gw, (gh//2):h:gh]
            points = np.stack((points_y.flatten(), points_x.flatten()), axis=-1)
            attn_plot = attn.cpu().numpy().flatten()

            # for yj in range(grid_size[0]):
            #     for xj in range(grid_size[1]):
            #         attn_plot[(yj * gh):(yj + 1) * gh, (xj * gw):(xj + 1) * gw] = attn[yj, xj].item()
            # attn_plot = np.minimum(attn_plot, 1) * 0.80
            attn_plot = attn_plot - np.min(attn_plot)
            attn_plot = attn_plot / np.max(attn_plot)
            attn_plot = scipy.interpolate.griddata(points, attn_plot, (grid_y, grid_x), method='linear', fill_value=0.2)
            attn_plot = np.float32(attn_plot)
            # heatmap = cv2.applyColorMap(np.uint8(255 * attn_plot), cv2.COLORMAP_BONE)
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # heatmap = np.float32(heatmap) / 255
            # attn_im = cv2.addWeighted(src_view, 0.5, heatmap, 0.5, 0)
            attn_im = cv2.cvtColor(attn_plot + 0.2, cv2.COLOR_GRAY2BGR) * src_view
            attn_im = np.uint8(255 * attn_im)

            ax = fig.add_subplot(gs[0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(attn_im)
            plt.savefig(os.path.join(out_dir, f"{yi}_{xi}.jpg"), bbox_inches='tight')
            plt.close(fig)

            if x_pos is not None and y_pos is not None:
                break
        if x_pos is not None and y_pos is not None:
            break

    # for i in tqdm.tqdm(range(attention_weights.shape[1])):
    #     xi = i % grid_size[1]
    #     yi = i // grid_size[0]
    #
    #     # if xi < 10 or xi > 24 or yi < 10 or yi > 24:
    #     #     continue
    #
    #     _ref_view = cv2.rectangle(ref_view.copy(), (xi * gw, yi * gh), ((xi + 1) * gw, (yi + 1) * gh),
    #                               (1, 0, 0))
    #     heads = attention_weights.shape[0]
    #     fig = plt.figure(figsize=(20, 40))
    #     gs = gridspec.GridSpec(1, 2)
    #     gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    #
    #     ax = fig.add_subplot(gs[0, 0])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(_ref_view)
    #
    #     _, min_indices = attention_heads_fused.topk(
    #         int(attention_heads_fused.size(-1) * discard_ratio), -1, False
    #     )
    #     attention_heads_fused = attention_heads_fused.scatter(1, min_indices, 0)
    #
    #     attn = np.zeros((sw, sh), np.float32)
    #     # avg_attention_weights = attention_weights.sum(dim=0) / heads
    #     # avg_attention_weights = avg_attention_weights[i] # - avg_attention_weights[i].min()
    #     # avg_attention_weights /= avg_attention_weights.max()
    #     for j in range(attention_weights.shape[1]):
    #         xj = j % grid_size[1]
    #         yj = j // grid_size[0]
    #         attn[(yj * gh):(yj + 1) * gh, (xj * gw):(xj + 1) * gw] = attention_heads_fused[i, j].item()
    #     # attn = np.minimum(attn, 1) * 0.80
    #     attn = attn - np.min(attn)
    #     attn = attn / np.max(attn)
    #
    #     heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #     heatmap = np.float32(heatmap) / 255
    #     attn_im = cv2.addWeighted(src_view, 0.5, heatmap, 0.5, 0)
    #     attn_im = np.uint8(255 * attn_im)
    #
    #     ax = fig.add_subplot(gs[0, 1])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(attn_im)
    #
    #     # heads_attn = attention_weights[:, i]
    #     # # heads_attn -= heads_attn.min()
    #     # # heads_attn /= heads_attn.max()
    #     #
    #     # for h in range(heads):
    #     #     sample_points_attn = heads_attn[h].numpy()
    #     #     grid = np.mgrid[0:1:(grid_size[1] * 1j), 0:1:(grid_size[0] * 1j)]
    #     #     grid[0] = grid[0] * (grid_size[1] - 1) / grid_size[1] + 1 / (2 * grid_size[1])
    #     #     grid[1] = grid[1] * (grid_size[0] - 1) / grid_size[0] + 1 / (2 * grid_size[0])
    #     #     grid = grid.reshape((2, grid_size[1] * grid_size[0])).T
    #     #     grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
    #     #     interpolation = griddata(grid, sample_points_attn, (grid_x, grid_y), method='cubic', fill_value=0)
    #     #     hx = 2 + (h // 4)
    #     #     hy = h % 4
    #     #     ax = fig.add_subplot(gs[hy, hx])
    #     #     ax.set_xticks([])
    #     #     ax.set_yticks([])
    #     #     ax.imshow(interpolation, cmap='gray', vmin=0., vmax=1.)
    #
    #     plt.savefig(os.path.join(out_dir, f"{i}.jpg"), bbox_inches='tight')
    #     plt.close(fig)


def inference_mv_detector(model, v0_path, v1_path, out_path, x_pos=None, y_pos=None):
    assert (x_pos is None and y_pos is None) or (x_pos is not None and y_pos is not None)
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # add information into dict
    data = dict(img_info=(dict(filename=v0_path), dict(filename=v1_path)), img_prefix=None)
    # build the data pipeline
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        _, attention = model.backbone.forward(data['img'][0], with_attn_weights=True)
        # Attention shape: B x V x (V-1) x H x W x H x W or  B x V x H x W x H x W

        for v in range(2):
            path_attention_dir = os.path.join(out_path, f"attention{v}")
            os.makedirs(path_attention_dir, exist_ok=True)
            h, w, _ = data['img_metas'][0][0]['img_shape'][v]
            img_show = data['img'][0][0, v, :, :h, :w].permute(1, 2, 0).cpu().numpy()
            img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

            ori_h, ori_w = data['img_metas'][0][0]['ori_shape'][v][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            model.show_result(
                img_show,
                result[v],
                out_file=out_path,
                score_thr=0.3)
            for _v in range(2):
                if v == _v:
                    continue
                ref_view = data['img'][0][0, v, ...]
                src_view = data['img'][0][0, _v, ...]
                ref_view -= ref_view.min()
                ref_view /= ref_view.max()
                src_view -= src_view.min()
                src_view /= src_view.max()
                if model.backbone.multiview_decoder_mode == "add":
                    attn = attention[0, v, 0]  # TODO: Change
                else:
                    attn = attention[0, v]
                draw_attention(ref_view, src_view, attn, path_attention_dir, x_pos=x_pos, y_pos=y_pos)

    return result


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    inference_mv_detector(model, *args.views_paths, args.output_dir, args.x, args.y)


if __name__ == '__main__':
    main()
