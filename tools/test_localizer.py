import argparse
from terminaltables import AsciiTable
import numpy as np
import pandas as pd
import torch

import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_localizer, localizers
from mmaction.models.tenons.segmental_consensuses import parse_stage_config
from mmaction.core.evaluation.localize_utils import (results2det,
                                                     perform_regression,
                                                     temporal_nms,
                                                     eval_ap_parallel,
                                                     det2df)


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--eval', type=str,
                        choices=['activitynet', 'thumos14'], help='eval types')
    parser.add_argument('--no_regression', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # reorganize stpp
    num_classes = (cfg.model.cls_head.num_classes -
                   1 if cfg.model.cls_head.with_bg
                   else cfg.model.cls_head.num_classes)
    stpp_feat_multiplier = 0
    for stpp_subcfg in cfg.model.segmental_consensus.stpp_cfg:
        _, mult = parse_stage_config(stpp_subcfg)
        stpp_feat_multiplier += mult
    cfg.model.segmental_consensus = dict(
        type="STPPReorganized",
        standalong_classifier=cfg.model.
        segmental_consensus.standalong_classifier,
        feat_dim=num_classes + 1 + num_classes * 3 * stpp_feat_multiplier,
        act_score_len=num_classes + 1,
        comp_score_len=num_classes,
        reg_score_len=num_classes * 2,
        stpp_cfg=cfg.model.segmental_consensus.stpp_cfg)

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_localizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint, strict=True)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(localizers, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

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
                for k, v in detections[cls].items()}
        print("NMS finished")

        if eval_type == 'activitynet':
            iou_range = np.arange(0.5, 1.0, 0.05)
        elif eval_type == 'thumos14':
            iou_range = np.arange(0.1, 1.0, .1)

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
