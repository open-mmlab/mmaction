import os
import os.path as osp
import logging
import mmcv
import time
import torch
import numpy as np
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset

from mmaction import datasets
from .accuracy import top_k_accuracy
from .ava_utils import (results2csv, read_csv, read_labelmap,
                        read_exclusions)

try:
    import sys
    sys.path.append(
        osp.abspath(osp.join(__file__, '../../../',
                             'third_party/ActivityNet/Evaluation/ava')))
    from mmaction.third_party.ActivityNet.Evaluation.ava import (
        object_detection_evaluation as det_eval)
    import standard_fields
except ImportError:
    print('Failed to import ActivityNet evaluation toolbox. Did you clone with'
          '"--recursive"?')


class DistEvalHook(Hook):
    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalTopKAccuracyHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 k=(1,)):
        super(DistEvalTopKAccuracyHook, self).__init__(dataset)
        self.k = k

    def evaluate(self, runner, results):
        gt_labels = []
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            gt_labels.append(ann['label'])

        results = [res.squeeze() for res in results]
        top1, top5 = top_k_accuracy(results, gt_labels, k=self.k)
        runner.mode = 'val'
        runner.log_buffer.output['top1 acc'] = top1
        runner.log_buffer.output['top5 acc'] = top5
        runner.log_buffer.ready = True


class AVADistEvalmAPHook(DistEvalHook):

    def __init__(self, dataset):
        super(AVADistEvalmAPHook, self).__init__(dataset)

    def evaluate(self, runner, results, verbose=False):

        categories, class_whitelist = read_labelmap(
            open(self.dataset.label_file))
        if verbose:
            logging.info("CATEGORIES ({}):\n".format(len(categories)))

        excluded_keys = read_exclusions(open(self.dataset.exclude_file))
        pascal_evaluator = det_eval.PascalDetectionEvaluator(
            categories)

        def print_time(message, start):
            logging.info("==> %g seconds to %s", time.time() - start, message)

        # Reads the ground truth data.
        boxes, labels, _ = read_csv(
            open(self.dataset.ann_file), class_whitelist, 0)
        start = time.time()
        for image_key in boxes:
            if verbose and image_key in excluded_keys:
                logging.info("Found excluded timestamp in detections: %s."
                             "It will be ignored.", image_key)
                continue
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                        np.array(labels[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                        np.zeros(len(boxes[image_key]), dtype=bool)
                })
        if verbose:
            print_time("Convert groundtruth", start)

        # Read detections datas.
        tmp_file = osp.join(runner.work_dir, 'temp_0.csv')
        results2csv(self.dataset, results, tmp_file)

        boxes, labels, scores = read_csv(open(tmp_file), class_whitelist, 50)
        start = time.time()
        for image_key in boxes:
            if verbose and image_key in excluded_keys:
                logging.info("Found excluded timestamp in detections: %s."
                             "It will be ignored.", image_key)
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                        np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                        np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                        np.array(scores[image_key], dtype=float)
                })
        if verbose:
            print_time("convert detections", start)

        start = time.time()
        metrics = pascal_evaluator.evaluate()
        if verbose:
            print_time("run_evaluator", start)
        for display_name in metrics:
            runner.log_buffer.output[display_name] = metrics[display_name]
        runner.log_buffer.ready = True
