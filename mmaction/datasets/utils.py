import copy
from collections import Sequence
import torch
import numpy as np
import os
import glob
import fnmatch
import mmcv
from mmcv.runner import obj_from_dict
from .. import datasets
import csv
import random


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def get_untrimmed_dataset(data_cfg):
    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']]
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)

    if len(dsets) > 1:
        raise ValueError("Not implemented yet")
    else:
        dset = dsets[0]

    return dset


def get_trimmed_dataset(data_cfg):
    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']]
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)

    if len(dsets) > 1:
        raise ValueError("Not implemented yet")
    else:
        dset = dsets[0]

    return dset


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.
    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".
    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def load_localize_proposal_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(info):
        offset = 0
        vid = info[offset]
        offset += 1

        n_frame = int(float(info[1]) * float(info[2]))
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset: offset + n_gt]]
        offset += n_gt
        n_pr = int(info[offset])
        offset += 1
        pr_boxes = [x.split() for x in info[offset: offset + n_pr]]

        return vid, n_frame, gt_boxes, pr_boxes

    return [parse_group(l) for l in info_list]


def process_localize_proposal_list(norm_proposal_list,
                                   out_list_name, frame_dict):
    norm_proposals = load_localize_proposal_file(norm_proposal_list)

    processed_proposal_list = []
    for idx, prop in enumerate(norm_proposals):
        vid = prop[0]
        frame_info = frame_dict[vid]
        frame_cnt = frame_info[1]
        frame_path = frame_info[0].split('/')[-1]

        gt = [[int(x[0]), int(float(x[1]) * frame_cnt),
               int(float(x[2]) * frame_cnt)] for x in prop[2]]

        prop = [[int(x[0]), float(x[1]), float(x[2]),
                 int(float(x[3]) * frame_cnt), int(float(x[4]) * frame_cnt)]
                for x in prop[3]]

        out_tmpl = "# {idx}\n{path}\n{fc}\n1\n{num_gt}\n{gt}{num_prop}\n{prop}"

        gt_dump = '\n'.join(['{} {:d} {:d}'.format(*x)
                             for x in gt]) + ('\n' if len(gt) else '')
        prop_dump = '\n'.join(['{} {:.04f} {:.04f} {:d} {:d}'.format(
            *x) for x in prop]) + ('\n' if len(prop) else '')

        processed_proposal_list.append(out_tmpl.format(
            idx=idx, path=frame_path, fc=frame_cnt,
            num_gt=len(gt), gt=gt_dump,
            num_prop=len(prop), prop=prop_dump))

    open(out_list_name, 'w').writelines(processed_proposal_list)


def parse_directory(path, key_func=lambda x: x[-11:],
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError(
                'x and y direction have different number '
                'of flow images. video: ' + f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict


def build_split_list(split, frame_info, shuffle=False):

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            if item[0] not in frame_info:
                # print("item:", item)
                continue
            elif frame_info[item[0]][1] > 0:
                rgb_cnt = frame_info[item[0]][1]
                flow_cnt = frame_info[item[0]][2]
                rgb_list.append('{} {} {}\n'.format(
                    item[0], rgb_cnt, item[1]))
                flow_list.append('{} {} {}\n'.format(
                    item[0], flow_cnt, item[1]))
            else:
                rgb_list.append('{} {}\n'.format(
                    item[0], item[1]))
                flow_list.append('{} {}\n'.format(
                    item[0], item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


def mimic_ucf101(frame_path, anno_dir):
    # Create classInd.txt, trainlist01.txt, testlist01.txt as in UCF101

    classes_list = os.listdir(frame_path)
    classes_list.sort()

    classDict = {}
    classIndFile = os.path.join(anno_dir, 'classInd.txt')
    with open(classIndFile, 'w') as f:
        for class_id, class_name in enumerate(classes_list):
            classDict[class_name] = class_id
            cur_line = str(class_id + 1) + ' ' + class_name + '\r\n'
            f.write(cur_line)


    for split_id in range(1, 4):
        splitTrainFile = os.path.join(anno_dir, 'trainlist%02d.txt' % (split_id))
        with open(splitTrainFile, 'w') as target_train_f:
            for class_name in classDict.keys():
                fname = class_name + '_test_split%d.txt' % (split_id)
                fname_path = os.path.join(anno_dir, fname)
                source_f = open(fname_path, 'r')
                source_info = source_f.readlines()
                for _, source_line in enumerate(source_info):
                    cur_info = source_line.split(' ')
                    video_name = cur_info[0]
                    if cur_info[1] == '1':
                        target_line = class_name + '/' + video_name + ' ' + str(classDict[class_name] + 1) + '\n'
                        target_train_f.write(target_line)

        splitTestFile = os.path.join(anno_dir, 'testlist%02d.txt' % (split_id))
        with open(splitTestFile, 'w') as target_test_f:
            for class_name in classDict.keys():
                fname = class_name + '_test_split%d.txt' % (split_id)
                fname_path = os.path.join(anno_dir, fname)
                source_f = open(fname_path, 'r')
                source_info = source_f.readlines()
                for _, source_line in enumerate(source_info):
                    cur_info = source_line.split(' ')
                    video_name = cur_info[0]
                    if cur_info[1] == '2':
                        target_line = class_name + '/' + video_name + ' ' + str(classDict[class_name] + 1) + '\n'
                        target_test_f.write(target_line)


def parse_hmdb51_splits(level):
    
    mimic_ucf101('data/hmdb51/rawframes', 'data/hmdb51/annotations')

    class_ind = [x.strip().split()
                 for x in open('data/hmdb51/annotations/classInd.txt')]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = '/'.join(vid.split('/')[-level:])
        label = class_mapping[items[0].split('/')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(
            'data/hmdb51/annotations/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open(
            'data/hmdb51/annotations/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits


def parse_ucf101_splits(level):
    class_ind = [x.strip().split()
                 for x in open('data/ucf101/annotations/classInd.txt')]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = '/'.join(vid.split('/')[-level:])
        label = class_mapping[items[0].split('/')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(
            'data/ucf101/annotations/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open(
            'data/ucf101/annotations/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits


def parse_kinetics_splits(level):
    csv_reader = csv.reader(
        open('data/kinetics400/annotations/kinetics_train.csv'))
    # skip the first line
    next(csv_reader)

    def convert_label(s):
        return s.replace('"', '').replace(' ', '_')

    labels_sorted = sorted(
        set([convert_label(row[0]) for row in csv_reader]))
    class_mapping = {label: i for i, label in enumerate(labels_sorted)}

    def list2rec(x, test=False):
        if test:
            vid = '{}_{:06d}_{:06d}'.format(x[0], int(x[1]), int(x[2]))
            label = -1  # label unknown
            return vid, label
        else:
            vid = '{}_{:06d}_{:06d}'.format(x[1], int(x[2]), int(x[3]))
            if level == 2:
                vid = '{}/{}'.format(convert_label(x[0]), vid)
            else:
                assert level == 1
            label = class_mapping[convert_label(x[0])]
            return vid, label
        
    csv_reader = csv.reader(
        open('data/kinetics400/annotations/kinetics_train.csv'))
    next(csv_reader)
    train_list = [list2rec(x) for x in csv_reader]
    csv_reader = csv.reader(
        open('data/kinetics400/annotations/kinetics_val.csv'))
    next(csv_reader)
    val_list = [list2rec(x) for x in csv_reader]
    csv_reader = csv.reader(
        open('data/kinetics400/annotations/kinetics_test.csv'))
    next(csv_reader)
    test_list = [list2rec(x, test=True) for x in csv_reader]

    return ((train_list, val_list, test_list), )
