import argparse

import mmcv
import numpy as np
from mmcv.runner import obj_from_dict

from mmaction import datasets
from mmaction.core.evaluation.localize_utils import (results2det,
                                                     perform_regression,
                                                     temporal_nms,
                                                     eval_ap_parallel,
                                                     det2df)

import pandas as pd
from terminaltables import AsciiTable


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('outputs', nargs='+')
    parser.add_argument('--eval', type=str,
                        choices=['activitynet', 'thumos14'], help='eval types')
    parser.add_argument('--no_regression', default=False, action='store_true')
    parser.add_argument('--score_weights', default=None, type=float, nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    output_list = []
    for out in args.outputs:
        output_list.append(mmcv.load(out))

    if args.score_weights:
        weights = np.array(args.score_weights) / sum(args.score_weights)
    else:
        weights = [1. / len(output_list) for _ in output_list]

    def merge_scores(idx):
        def merge_part(arrs, index, weights):
            if arrs[0][index] is not None:
                return np.sum([a[index] * w for a, w in zip(arrs, weights)],
                              axis=0)
            else:
                return None

        results = [output[idx] for output in output_list]
        rel_props = output_list[0][idx][0]
        return (rel_props, merge_part(results, 1, weights),
                merge_part(results, 2, weights),
                merge_part(results, 3, weights))

    print('Merge detection scores from {} sources'.format(len(output_list)))
    outputs = [merge_scores(idx) for idx in range(len(dataset))]
    print('Merge finished')

    eval_type = args.eval
    if eval_type:
        print('Starting evaluate {}'.format(eval_type))

        detections = results2det(
            dataset, outputs, **cfg.test_cfg.ssn.evaluater)

        if not args.no_regression:
            print("Performing location regression")
            for cls in range(len(detections)):
                detections[cls] = {
                    k: perform_regression(v)
                    for k, v in detections[cls].items()
                }
            print("Regression finished")

        print("Performing NMS")
        for cls in range(len(detections)):
            detections[cls] = {
                k: temporal_nms(v, cfg.test_cfg.ssn.evaluater.nms)
                for k, v in detections[cls].items()
            }
        print("NMS finished")

        if eval_type == 'activitynet':
            iou_range = np.arange(0.5, 1.0, 0.05)
        elif eval_type == 'thumos14':
            iou_range = np.arange(0.1, 1.0, .1)
            # iou_range = [0.5]

        # get gt
        all_gt = pd.DataFrame(dataset.get_all_gt(), columns=[
                              'video-id', 'cls', 't-start', 't-end'])
        gt_by_cls = [all_gt[all_gt.cls == cls].reset_index(
            drop=True).drop('cls', 1)
            for cls in range(len(detections))]
        plain_detections = [det2df(detections, cls)
                            for cls in range(len(detections))]

        ap_values = eval_ap_parallel(plain_detections, gt_by_cls, iou_range)
        map_iou = ap_values.mean(axis=0)
        print("Evaluation finished")

        # display
        display_title = 'Temporal detection performance ({})'.format(args.eval)
        display_data = [['IoU thresh'], ['mean AP']]

        for i in range(len(iou_range)):
            display_data[0].append('{:.02f}'.format(iou_range[i]))
            display_data[1].append('{:.04f}'.format(map_iou[i]))
        table = AsciiTable(display_data, display_title)
        table.justify_columns[-1] = 'right'
        table.inner_footing_row_border = True
        print(table.table)


if __name__ == '__main__':
    main()
