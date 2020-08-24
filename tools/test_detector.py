import argparse
import time
import os.path as osp
import tempfile

import torch
import torch.distributed as dist
import mmcv
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.runner import get_dist_info
from mmcv.parallel.distributed import MMDistributedDataParallel

from mmaction import datasets
from mmaction.apis import init_dist
from mmaction.datasets import build_dataloader
from mmaction.models import build_detector, detectors
from mmaction.core.evaluation.ava_utils import results2csv, ava_eval

import os.path as osp


args = None


def multiple_test(model, data_loader, tmpdir='./tmp'):
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
            result = model(return_loss=False, rescale=True, **data)
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


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results



def parse_args():
    parser = argparse.ArgumentParser(description='Test an action detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpus', default=8, type=int, help='GPU number used for testing')
    parser.add_argument('--out', help='output result file', default='detection_result.pkl')
    parser.add_argument('--eval', type=str,
                        choices=['proposal', 'bbox'], help='eval types')
    parser.add_argument('--ann_file', type=str,
                        default='data/ava/annotations/ava_val_v2.1.csv')
    parser.add_argument('--label_file', type=str,
                        default='data/ava/annotations/'
                        'ava_action_list_v2.1_for_activitynet_2018.pbtxt')
    parser.add_argument('--exclude_file', type=str,
                        default='data/ava/annotations/'
                        'ava_val_excluded_timestamps_v2.1.csv')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    global args
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if args.out is None or not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if osp.exists(args.out):
        outputs = mmcv.load(args.out)
    else:   
        if args.launcher == 'none':
          raise NotImplementedError("By default, we use distributed testing, so that launcher should be pytorch")
        else:
          distributed = True
          init_dist(args.launcher, **cfg.dist_params)

        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
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

        outputs = multiple_test(model, data_loader)
    

        rank, _ = get_dist_info()
        if rank == 0:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)

    eval_type = args.eval
    if eval_type:
        print('Starting evaluate {}'.format(eval_type))

        result_file = osp.join(args.out + '.csv')
        results2csv(dataset, outputs, result_file)

        ava_eval(result_file, eval_type,
                 args.label_file, args.ann_file, args.exclude_file)


if __name__ == '__main__':
    main()
