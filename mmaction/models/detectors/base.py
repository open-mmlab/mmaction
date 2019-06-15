import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn

from mmaction.core import get_classes
from mmaction.utils.misc import tensor2video_snaps


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @abstractmethod
    def extract_feat(self, img_group):
        pass

    def extract_feats(self, img_groups):
        assert isinstance(img_groups, list)
        for img_group in img_groups:
            yield self.extract_feat(img_group)

    @abstractmethod
    def forward_train(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, num_modalities, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, num_modalities, img_metas, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError('{} must be a list, but got {}'.format(
                img_metas, type(img_metas)))

        num_augs = len(kwargs['img_group_0'])
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    num_augs, len(img_metas)))
        # TODO: remove the restriction of videos_per_gpu == 1 when prepared
        videos_per_gpu = kwargs['img_group_0'][0].size(0)
        assert videos_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(num_modalities, img_metas, **kwargs)
        else:
            return self.aug_test(num_modalities, img_metas, **kwargs)

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)

    def show_result(self,
                    data,
                    bbox_result,
                    img_norm_cfg,
                    dataset='ava',
                    score_thr=0.3):

        img_group_tensor = data['img_group_0'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2video_snaps(img_group_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)
