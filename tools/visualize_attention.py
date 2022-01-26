import argparse
import os

import cv2
import mmcv
import numpy as np
import scipy
import torch
import tqdm
from matplotlib import pyplot as plt, gridspec
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from scipy.interpolate import griddata

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


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
    ref_view = ref_view.clone().permute(1, 2, 0).cpu().numpy()
    src_view = src_view.clone().permute(1, 2, 0).cpu().numpy()

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

            grid_x, grid_y = np.mgrid[0:w, 0:h]
            points_x, points_y = np.mgrid[(gw//2):w:gw, (gh//2):h:gh]
            points = np.stack((points_y.flatten(), points_x.flatten()), axis=-1)
            attn_plot = attn.cpu().numpy().flatten()

            attn_plot = attn_plot - np.min(attn_plot)
            attn_plot = attn_plot / np.max(attn_plot)
            attn_plot = scipy.interpolate.griddata(points, attn_plot, (grid_y, grid_x), method='linear', fill_value=0.2)
            attn_plot = np.float32(attn_plot)
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

    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    inference_mv_detector(model, *args.views_paths, args.output_dir, args.x, args.y)


if __name__ == '__main__':
    main()
