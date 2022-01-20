import argparse
import json

from mmdet.datasets.api_wrappers import COCOeval, COCO


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
    # all_data = [json.load(open(data, 'r')) for data in all_data]
    # all_results = [json.load(open(result, 'r')) for result in all_results]

    modp = 0
    frames = 0
    for data, results in zip(all_data, all_results):
        imgs = data['images']
        anns = data['annotations']
        gts = {im['id']: [ann for ann in anns if ann['image_id'] == im['id']] for im in imgs}
        results = {im['id']: [r for r in results if r['image_id'] == im['id']] for im in imgs}

        for k in gts.keys():
            i_gt = gts[k]  # lists of objects in the ith frame
            i_result = results[k]  # list of results of the ith frame

            overlap_ratio = 0
            n_mapped = 0
            for obj in i_gt:
                category = obj['category_id']
                box = obj['bbox']
                COCOeval()
                ious = [get_iou(box, b['bbox']) if category == b['category_id'] else 0 for b in i_result]
                if len(ious) > 0:
                    max_iou = max(ious)
                    if max_iou > th:
                        overlap_ratio += max_iou
                        n_mapped += 1
            if n_mapped > 0:
                modp += overlap_ratio / n_mapped
        frames += len(gts.keys())
    modp /= frames
    print(f"MODP: {modp}")


if __name__ == '__main__':
    main()