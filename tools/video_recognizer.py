#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   video_recognizer.py.py    
@Contact :   juzheng@hxdi.com
@License :   (C)Copyright 2018-2019, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/9/3 11:14   juzheng      1.0         None
"""

import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import os
from mmcv.parallel import DataContainer as DC
from mmcv.visualization import color_val
from mmaction.datasets.transforms import ImageTransform, GroupImageTransform

from mmaction.models import build_recognizer
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
from collections import Sequence
from mmcv.parallel import MMDataParallel
import cv2


class RawFramesDataset():
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 div_255=False,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 test_mode=False,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW'):
        # prefix of images path
        self.img_prefix = img_prefix

        # normalization config
        self.img_norm_cfg = img_norm_cfg

        # parameters for frame fetching
        # number of segments
        self.num_segments = num_segments
        # number of consecutive frames
        self.old_length = new_length * new_step
        self.new_length = new_length
        # number of steps (sparse sampling for efficiency of io)
        self.new_step = new_step
        # whether to temporally random shift when training
        self.random_shift = random_shift
        # whether to temporally jitter if new_step > 1
        self.temporal_jitter = temporal_jitter

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            self.modalities = modality
            num_modality = len(modality)
        else:
            self.modalities = [modality]
            num_modality = 1
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
        if img_scale_file is not None:
            self.img_scale_dict = {line.split(' ')[0]:
                                   (int(line.split(' ')[1]),
                                    int(line.split(' ')[2]))
                                   for line in open(img_scale_file)}
        else:
            self.img_scale_dict = None
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
        self.div_255 = div_255

        # parameters for data augmentation
        # flip ratio
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        # transforms
        assert oversample in [None, 'three_crop', 'ten_crop']
        self.img_group_transform = GroupImageTransform(
            size_divisor=None, crop_size=self.input_size,
            oversample=oversample, random_crop=random_crop,
            more_fix_crop=more_fix_crop,
            multiscale_crop=multiscale_crop, scales=scales,
            max_distort=max_distort,
            **self.img_norm_cfg)

        # input format
        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format
        '''
        self.bbox_transform = Bbox_transform()
        '''

    def load_annotations(self, ann_file):
        return [x.strip().split(' ') for x in open(ann_file)]

    def get_data(self, img_group):

        # if self.test_mode:
        #     segment_indices, skip_offsets = self._get_test_indices(record)
        # else:
        #     segment_indices, skip_offsets = self._sample_indices(
        #         record) if self.random_shift else self._get_val_indices(record)

        data = dict(num_modalities=DC([[to_tensor(len(self.modalities))]]))

        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        # img_group = self._get_frames(
        #     record, image_tmpl, modality, segment_indices, skip_offsets)
        #
        flip = True if np.random.rand() < self.flip_ratio else False
        # if (self.img_scale_dict is not None
        #         and record.path in self.img_scale_dict):
        #     img_scale = self.img_scale_dict[record.path]
        # else:
        img_scale = self.img_scale
        (img_group, img_shape, pad_shape,
         scale_factor, crop_quadruple) = self.img_group_transform(
            img_group, img_scale,
            crop_history=None,
            flip=flip, keep_ratio=self.resize_keep_ratio,
            div_255=self.div_255,
            is_flow=True if modality == 'Flow' else False)
        ori_shape = (256, 340, 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            crop_quadruple=crop_quadruple,
            flip=flip)
        # [M x C x H x W]
        # M = 1 * N_oversample * N_seg * L

        data.update(dict(
            img_group_0=DC([to_tensor([img_group])], stack=True, pad_dims=2),
            img_meta=DC(img_meta, cpu_only=True)
        ))

        return data


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


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(**cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=(cfg.data.test.img_scale, cfg.data.test.img_scale),
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    # img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]

    return dict(num_modalities=DC([[to_tensor(1)]]), img_group_0=DC(to_tensor(img), stack=True, pad_dims=2),
                img_meta=DC(img_meta, cpu_only=True))


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
        print(data)
        break
    return results


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = mmcv.Config.fromfile('configs/ucf101/tsn_rgb_bninception.py')

    model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # model.to('cuda:0')
    # model.eval()
    # load_checkpoint(model, args.checkpoint, strict=True)
    model = MMDataParallel(model, device_ids=[0])
    model.cfg = cfg

    """
    图片识别
    """
    from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
    from mmaction import datasets
    from mmaction.datasets import build_dataloader
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    outputs = single_test(model, data_loader)

    use_softmax = False
    if use_softmax is True:
        print("Averaging score over {} clips with softmax".format(
            outputs[0].shape[0]))
        results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
    else:
        print("Averaging score over {} clips without softmax (ie, raw)".format(
            outputs[0].shape[0]))
        results = [res.mean(axis=0) for res in outputs]
    print("result:{}".format(results))
    pred = int(np.argmax(results, axis=1))
    print(pred)
    print(results[0][pred])
    ann_file = "data/ucf101/annotations/classInd.txt"
    label_name = [x.strip().split(' ') for x in open(ann_file)]
    print(label_name[pred])
    #
    """
    视频识别
    """
    # cap = cv2.VideoCapture('data/ucf101/videos/HandstandPushups/v_HandStandPushups_g01_c01.avi')
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     outputs = []
    #     args = cfg.data.test.copy()
    #     obj_type = args.pop('type')
    #     data = RawFramesDataset(**args)
    #     imgs = data.get_data([frame])
    #     gt_labels = data.load_annotations("data/ucf101/annotations/classInd.txt")
    #     with torch.no_grad():
    #         output = model(return_loss=False, **imgs)
    #         outputs.append(output)
    #
    #     use_softmax = True
    #     if use_softmax is True:
    #         print("Averaging score over {} clips with softmax".format(
    #             outputs[0].shape[0]))
    #         results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
    #     else:
    #         print("Averaging score over {} clips without softmax (ie, raw)".format(
    #             outputs[0].shape[0]))
    #         results = [res.mean(axis=0) for res in outputs]
    #     pred = int(np.argmax(results, axis=1))
    #     print(pred)
    #     print(results[0][pred])
    #     print(gt_labels[pred])
    #     text_color = color_val('green')
    #     cv2.putText(frame, gt_labels[pred][1], (100, 100),
    #                 cv2.FONT_HERSHEY_COMPLEX, 1, text_color)
    #     cv2.imshow('image', frame)
    #     cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


