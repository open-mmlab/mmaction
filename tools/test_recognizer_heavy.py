import argparse
import time
import torch
import torch.distributed as dist
import mmcv
import os
import os.path as osp
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.runner import get_dist_info
from mmcv.parallel.distributed import MMDistributedDataParallel
import tempfile
from mmaction import datasets
from mmaction.apis import init_dist
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer
from mmaction.core.evaluation.accuracy import softmax, top_k_accuracy
from mmaction.core.evaluation.accuracy import mean_class_accuracy
import warnings
from functools import reduce
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

        img_group = data['img_group_0'].data[0]

        totlen = img_group.size()[1]
        batch_data = []
        st = 0
        scores = []

        assert args.batch_size == -1 or totlen % args.batch_size == 0
        while st < totlen:
            if args.batch_size == -1:
                sz = totlen
            else:
                sz = args.batch_size if st + args.batch_size <= totlen else totlen - st

            ed = st + sz
            batch_data = img_group[:, st: ed].cuda()
            fake_data = {}
            fake_data['img_group_0'] = batch_data
            fake_data['num_modalities'] = data['num_modalities']
            fake_data['img_meta'] = data['img_meta']
            with torch.no_grad():
                score = model(return_loss=False, **fake_data)

            scores.append(score)

            st = ed

        result = reduce(lambda x, y: x+y, scores) / len(scores)
        results.append(result)

        toc = time.time()
        proc_time_pool = proc_time_pool + toc - tac

        tic = toc
    print('rank {}, begin collect results'.format(rank), flush=True)
    results = collect_results(results, len(data_loader.dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
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
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--testfile', help='checkpoint file', type=str, default='')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true', help='whether to use softmax score')
    parser.add_argument('--local_rank', type=int, default=0)
    # batch_size should be divided by total crops
    parser.add_argument('--batch_size', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # must use fcn testing
    cfg.model.update({'fcn_testing': True})
    cfg.model['cls_head'].update({'fcn_testing': True})

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if args.testfile != '':
        cfg.data.test.ann_file = args.testfile

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    assert distributed, "We only support distributed testing"

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
