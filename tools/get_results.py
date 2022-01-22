import argparse
import json

import numpy as np
import scipy.optimize


def get_iou(box1, box2):
    """
    Boxes are in the form [x y w h], where x and y are top left corners
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_b = max(min(x1 + w1, x2 + w2) - max(x1, x2), 0)
    inter_h = max(min(y1 + h1, y2 + h2) - max(y1, y2), 0)
    inter = inter_b * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def parse_args():
    parser = argparse.ArgumentParser(description='Gets MODA and MODP metrics from COCO and results files')
    parser.add_argument('--datasets', nargs="*", help='COCO datasets')
    parser.add_argument('--results', nargs="*", help='results files')
    parser.add_argument('--th', type=float, default=0.2)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    all_data = args.datasets
    all_results = args.results
    th = args.th
    assert len(all_data) == len(all_results)
    all_data = [json.load(open(data, 'r')) for data in all_data]
    all_results = [json.load(open(result, 'r')) for result in all_results]

    modp = 0
    misses = 0
    fps = 0
    ng = 0
    frames = 0
    for data, results in zip(all_data, all_results):
        imgs = data['images']
        anns = data['annotations']
        gts = {im['id']: [ann for ann in anns if ann['image_id'] == im['id']] for im in imgs}
        results = {im['id']: [r for r in results if r['image_id'] == im['id']] for im in imgs}

        for i_gt, i_result in zip(gts.values(), results.values()):
            # i_gt list of objects in the ith frame
            # i_result list of results of the ith frame
            ng += len(i_gt)
            if len(i_gt) > 0 and len(i_result) > 0:
                cost_matrix = [[get_iou(obj['bbox'], b['bbox']) if obj['category_id'] == b['category_id'] else 0
                                for b in i_result]
                               for obj in i_gt]  # Ng x Nd matrix
                cost_matrix = np.array(cost_matrix)
                row_id, col_id = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                overlap_ratio = cost_matrix[row_id, col_id].sum()
                n_mapped = len(row_id)
                if n_mapped > 0:
                    modp += overlap_ratio / len(row_id)

                gt_iou = cost_matrix.max(axis=1)
                misses_t = gt_iou <= th
                misses += misses_t.sum()

                det_iou = cost_matrix.max(axis=0)
                fps_t = det_iou <= th
                fps += fps_t.sum()

            elif ng > 0:  # no predictions, ng misses
                misses += ng
            else:  # ng = 0, no misses, false positives = predictions
                fps += len(i_result)

        frames += len(gts.keys())
    modp /= frames
    moda = 1 - (misses + fps) / ng
    print(f"MODA: {moda}")
    print(f"MODP: {modp}")


if __name__ == '__main__':
    main()
