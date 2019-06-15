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
        frame_path = frame_info[0]

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
                    flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = glob.glob(os.path.join(path, '*'))

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
