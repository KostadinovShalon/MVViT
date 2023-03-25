import argparse
import json
import tqdm
import os

views = 7


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the Wildtrack dataset to MV COCO format')
    parser.add_argument('--anns-dir', help='Root folder for the camera annotations')
    parser.add_argument('--output-dir', help='Output directory')
    args = parser.parse_args()
    return args


def create_annotations(root, ann_files):
    img_id = 1
    ann_id = 1
    images = [list() for _ in range(views)]  # Array of arrays of dicts
    annotations = [list() for _ in range(views)]  # Array of arrays of dicts

    for ann_file in tqdm.tqdm(ann_files):
        data = json.load(open(os.path.join(root, ann_file), 'r'))

        W, H = 1920, 1080

        image_name = ann_file.rsplit('.', 1)[0] + ".png"
        for v in range(views):
            images[v].append({
                "id": img_id,
                "file_name": f"C{v + 1}/{image_name}",
                "license": 1,
                "width": W,
                "height": H
            })
        for instance in data:
            for v in range(views):
                xmax, ymax, xmin, ymin = instance["views"][v]['xmax'], instance["views"][v]['ymax'], instance["views"][v][
                    'xmin'], instance["views"][v]['ymin']
                if not (xmax == -1 or ymax == -1 or xmin == -1 or ymin == 1):
                    x = xmin
                    y = ymin
                    w = xmax - xmin
                    h = ymax - ymin
                    annotations[v].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": [],
                        "area": w * h,
                        "bbox": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                        "iscrowd": 0
                    })
            ann_id += 1
        img_id += 1
    return images, annotations


def main():
    args = parse_args()
    anns_root = args.anns_dir
    output_dir = args.output_dir

    base_dict = {
        "info": {
            "year": 2021,
            "version": 1,
            "description": "WildTrack dataset",
            "contributor": "",
            "url": "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
            "date_created": "2021-08-27"
        },
        "licenses": [{
            "id": 1,
            "name": "GPL 2",
            "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
        }],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "Person"
            }
        ]
    }

    train_datasets = [dict(base_dict) for _ in range(views)]
    val_datasets = [dict(base_dict) for _ in range(views)]

    annotation_files = [file for file in os.listdir(anns_root) if
                        file.endswith(".json")]
    annotation_files.sort()

    total_train_files = int(0.8*len(annotation_files))
    train_annotation_files = annotation_files[:total_train_files]
    val_annotation_files = annotation_files[total_train_files:]

    output_train_files = [f"C{v + 1}_train.json" for v in range(views)]
    output_val_files = [f"C{v + 1}_val.json" for v in range(views)]

    train_images, train_annotations = create_annotations(anns_root, train_annotation_files)
    val_images, val_annotations = create_annotations(anns_root, val_annotation_files)

    for v, dataset in enumerate(train_datasets):
        dataset['images'] = train_images[v]
        dataset['annotations'] = train_annotations[v]
        json.dump(dataset, open(os.path.join(output_dir, output_train_files[v]), 'w'))

    for v, dataset in enumerate(val_datasets):
        dataset['images'] = val_images[v]
        dataset['annotations'] = val_annotations[v]
        json.dump(dataset, open(os.path.join(output_dir, output_val_files[v]), 'w'))


if __name__ == '__main__':
    main()
