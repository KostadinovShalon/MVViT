import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_color(index):
    if index % 7 == 0:
        return np.array([[1, 0, 0]])
    if index % 7 == 1:
        return np.array([[0, 1, 0]])
    if index % 7 == 2:
        return np.array([[0, 0, 1]])
    if index % 7 == 3:
        return np.array([[1, 1, 0]])
    if index % 7 == 4:
        return np.array([[0, 1, 1]])
    if index % 7 == 5:
        return np.array([[1, 0, 1]])
    if index % 7 == 6:
        return np.array([[0.5, 0.5, 0.5]])


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    # legend = args.legend
    # if legend is None:

    epochs = list(log_dicts[0].keys())
    common_layers = log_dicts[0][epochs[0]].keys()
    common_layers = [layer for layer in common_layers if layer.startswith(tuple(args.modules))]
    common_layers = [layer for layer in common_layers if not any(s in layer for s in args.exclude)]

    for ld in log_dicts[1:len(log_dicts)]:
        new_layers = ld[epochs[0]].keys()
        new_layers = [layer for layer in new_layers if layer.startswith(tuple(args.modules))]
        new_layers = [layer for layer in new_layers if not any(s in layer for s in args.exclude)]
        common_layers = [layer for layer in common_layers if layer in new_layers]

    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        print(f'plot curve of {args.json_logs[i]}, for {len(epochs)} epochs')
        gray_levels = np.array([[1 - 0.5 * e / len(epochs) for e in epochs]])
        colors = gray_levels.T @ get_color(i)
        for j, epoch in enumerate(epochs):

            xs = []
            ys = []
            for layer in list(common_layers):
                xs.append(layer)
                ys.append(log_dict[epoch][layer][-1])
            extra_layers = log_dict[epoch].keys()
            extra_layers = [layer for layer in extra_layers if layer.startswith(tuple(args.modules))]
            extra_layers = [layer for layer in extra_layers if not any(s in layer for s in args.exclude)]
            extra_layers = [layer for layer in extra_layers if layer not in common_layers]
            for layer in list(extra_layers):
                xs.append(layer)
                ys.append(log_dict[epoch][layer][-1])

            xs = np.array(xs)
            ys = np.array(ys)
            plt.xlabel('layer')
            plt.xticks(rotation=90)
            plot_args = dict(linewidth=0.5, color=tuple(colors[j]))
            if j == 0 and args.legend is not None:
                plot_args['label'] = args.legend[i]
            plt.plot(
                xs, ys, **plot_args)
            plt.legend()
    if args.title is not None:
        plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--modules',
        type=str,
        nargs='+',
        default=['backbone', 'neck', 'bbox_head'],
        help='list of modules to be included')
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=['bias'],
        help='list of modules to be excluded')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()
