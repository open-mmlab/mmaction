import argparse

import torch
import time
import torch.distributed as dist
import mmcv
import os.path as osp
import tempfile
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.runner import get_dist_info
from mmcv.parallel.distributed import MMDistributedDataParallel

from mmaction import datasets
from mmaction.apis import init_dist
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
args = None


def multi_test(model, data_loader, tmpdir='./tmp'):
    global args
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    count = 0
    data_time_pool = 0
    proc_time_pool = 0
    tic = time.time()
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print('rank {}, data_batch {}'.format(rank, i))
        count = count + 1
        tac = time.time()
        data_time_pool = data_time_pool + tac - tic

        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc
    print('rank {}, begin collect results'.format(rank), flush=True)
    results = collect_results(results, len(data_loader.dataset), tmpdir)
    return results


def collect_results(result_part, size, tmpdir=None):
    global args

    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        tmpdir = osp.join(tmpdir, args.out.split('.')[0])
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir

    print('rank {} begin dump'.format(rank), flush=True)
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    print('rank {} finished dump'.format(rank), flush=True)
    dist.barrier()
    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--out', help='output result file', default='default.pkl')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    # only for TSN3D
    parser.add_argument('--fcn_testing', action='store_true',
                        help='use fcn testing for 3D convnet')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # pass arg of fcn testing
    if args.fcn_testing:
        cfg.model.update({'fcn_testing': True})
        cfg.model['cls_head'].update({'fcn_testing': True})

    # for regular testing
    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if args.launcher == 'none':
        raise NotImplementedError("By default, we use distributed testing, so that launcher should be pytorch")
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    outputs = multi_test(model, data_loader)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

        gt_labels = []
        for i in range(len(dataset)):
            ann = dataset.get_ann_info(i)
            gt_labels.append(ann['label'])

        if args.use_softmax:
            print("Averaging score over {} clips with softmax".format(outputs[0].shape[0]))
            results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
        else:
            print("Averaging score over {} clips without softmax (ie, raw)".format(outputs[0].shape[0]))
            results = [res.mean(axis=0) for res in outputs]
        top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
        mean_acc = mean_class_accuracy(results, gt_labels)
        print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
        print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
        print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
    main()
