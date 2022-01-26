import argparse
import itertools
import json
import os

import mmcv
import numpy as np
import torch
from mmcv import Config, print_log
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import logging
from terminaltables import AsciiTable

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint')
    parser.add_argument('--classwise', action='store_true')
    parser.add_argument('--combined_gt')
    parser.add_argument('--combined_results')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    args = parser.parse_args()
    return args


def combine_gts_and_results(gts_path, result_files, out_dir):
    # both are lists
    data = [json.load(open(filepath, 'r')) for filepath in gts_path]
    results = [json.load(open(rf, 'r')) for rf in result_files]

    out_data = data[0].copy()
    out_results = results[0].copy()

    for v_data, r_data in zip(data[1:], results[1:]):
        last_img_id = max([im['id'] for im in out_data["images"]])
        last_ann_id = max([ann['id'] for ann in out_data["annotations"]])
        for img in v_data['images']:
            img['id'] += last_img_id
            out_data['images'].append(img)
        for ann in v_data['annotations']:
            ann['id'] += last_ann_id
            ann['image_id'] += last_img_id
            out_data['annotations'].append(ann)
        for r in r_data:
            r['image_id'] += last_img_id
            out_results.append(r)
    combined_results_path = os.path.join(out_dir, "combined_results.json")
    combined_gts_path = os.path.join(out_dir, "combined_gts.json")
    json.dump(out_results, open(combined_results_path, 'w'))
    json.dump(out_data, open(combined_gts_path, 'w'))
    return combined_gts_path, combined_results_path


def create_combined_files(args):
    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = int(len(result) / dataset.views)
        results.extend(result)  # BV results

        for _ in range(batch_size):
            prog_bar.update()

    views = len(dataset.cocos)
    result_files = []
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    for view in range(views):
        sv_results = [results[i * views + view] for i in range(len(dataset))]
        result_file, tmp_dir = dataset.format_results(sv_results, os.path.join(args.output_dir, f"v{view}"))
        result_files.append(result_file['bbox'])

    ann_files = dataset.ann_files
    combined_gt, combined_results = combine_gts_and_results(ann_files, result_files, args.output_dir)
    return combined_gt, combined_results


def main():
    args = parse_args()
    if args.combined_gt is None or args.combined_results is None:
        combined_gt, combined_results = create_combined_files(args)
    else:
        combined_gt = args.combined_gt
        combined_results = args.combined_results
    cocoGt = COCO(combined_gt)
    msg = f'\nEvaluating bbox...'
    print_log(msg, logger=None)
    try:
        cocoDt = cocoGt.loadRes(combined_results)
    except IndexError:
        print_log(
            'The testing results of the whole dataset is empty.',
            logger=None,
            level=logging.ERROR)
        return

    iou_type = 'bbox'
    cat_ids = cocoGt.get_cat_ids()
    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.params.catIds = cat_ids
    cocoEval.params.imgIds = cocoGt.get_img_ids()
    cocoEval.params.maxDets = list((100, 300, 1000))
    cocoEval.params.iouThrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    # mapping of cocoEval.stats
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if args.classwise:  # Compute per-category AP
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = cocoEval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = cocoGt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=None)

    metric_items = [
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'
    ]
    eval_results = {}
    for metric_item in metric_items:
        key = f'bbox_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results[f'bbox_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        f'{ap[4]:.3f} {ap[5]:.3f}')

    eval_results['bbox_mAP_copypaste'] = ' '.join([str(v) for v in eval_results.values()])
    print(eval_results)


if __name__ == '__main__':
    main()
