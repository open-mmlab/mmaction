import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (GroupImageTransform)
from .utils import to_tensor

import lmdb


class RawFramesRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class LMDBFramesDataset(Dataset):
    def __init__(self,
                 ann_file,
                 lmdb_prefix,
                 img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 modality='RGB',
                 lmdb_tmpl='{}_img_lmdb',
                 key_tmpl='img_{:05d}',
                 img_scale=256,
                 input_size=224,
                 div_255=False,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 test_mode=False,
                 oversample=False,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW'):
        # prefix of lmdb path
        self.lmdb_prefix = lmdb_prefix

        # load annotations
        self.video_infos = self.load_annotations(ann_file)

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

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            self.modalities = modality
            num_modality = len(modality)
        else:
            self.modalities = [modality]
            num_modality = 1
        self.lmdb_tmpl = lmdb_tmpl
        if isinstance(key_tmpl, (list, tuple)):
            self.key_tmpls = key_tmpl
        else:
            self.key_tmpls = [key_tmpl]
        assert len(self.key_tmpls) == num_modality

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
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

        # test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
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

    def __len__(self):
        return len(self.video_infos)

    def load_annotations(self, ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]
        # return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return {'path': self.video_infos[idx].path,
                'num_frames': self.video_infos[idx].num_frames,
                'label': self.video_infos[idx].label}
        # return self.video_infos[idx]['ann']

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.img_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1

    def _load_image(self, key_tmpl, modality, idx):
        print("key_tmpl.format(idx)=", key_tmpl.format(idx))
        if modality in ['RGB', 'RGBDiff']:
            img_bytes = self.txn.get(str(key_tmpl.format(idx)).encode())
            return [mmcv.imfrombytes(img_bytes, flag='color')]
        elif modality == 'Flow':
            img_bytes = self.txn.get(key_tmpl.format('x', idx))
            x_imgs = mmcv.imfrombytes(img_bytes, flag='grayscale')
            img_bytes = self.txn.get(key_tmpl.format('y', idx))
            y_imgs = mmcv.imfrombytes(img_bytes, flag='grayscale')
            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality'
                'should be ["RGB", "RGBDiff", "Flow"]')

    def _sample_indices(self, record):
        '''

        :param record: VideoRawFramesRecord
        :return: list, list
        '''
        average_duration = (record.num_frames -
                            self.old_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif record.num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(np.random.randint(
                record.num_frames - self.old_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)
        return offsets + 1, skip_offsets  # frame index starts from 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)
        return offsets + 1, skip_offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)
        return offsets + 1, skip_offsets

    def _get_frames(self, record, key_tmpl, modality, indices, skip_offsets):

        self.env = lmdb.open(
            osp.join(self.lmdb_prefix, self.lmdb_tmpl.format(record.path)),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.txn = self.env.begin(write=False)

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                seg_imgs = self._load_image(
                    key_tmpl, modality, p + skip_offsets[i])
                images.extend(seg_imgs)
                if p + self.new_step < record.num_frames:
                    p += self.new_step
        return images

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))),
                    gt_label=DC(to_tensor(record.label),
                                stack=True, pad_dims=None))

        # handle the first modality
        modality = self.modalities[0]
        key_tmpl = self.key_tmpls[0]
        img_group = self._get_frames(
            record, key_tmpl, modality, segment_indices, skip_offsets)

        flip = True if np.random.rand() < self.flip_ratio else False
        (img_group, img_shape, pad_shape,
         scale_factor, crop_quadruple) = self.img_group_transform(
            img_group, self.img_scale,
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
        if self.input_format == "NCTHW":
            img_group = img_group.reshape(
                (-1, self.num_segments, self.new_length) + img_group.shape[1:])
            # N_over x N_seg x L x C x H x W
            img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
            # N_over x N_seg x C x L x H x W
            img_group = img_group.reshape((-1,) + img_group.shape[2:])
            # M' x C x L x H x W

        data.update(dict(
            img_group_0=DC(to_tensor(img_group), stack=True, pad_dims=2),
            img_meta=DC(img_meta, cpu_only=True)
        ))

        # handle the rest modalities using the same
        for i, (modality, key_tmpl) in enumerate(
                zip(self.modalities[1:], self.key_tmpls[1:])):
            img_group = self._get_frames(
                record, key_tmpl, modality, segment_indices, skip_offsets)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            (img_group, img_shape, pad_shape,
             scale_factor, crop_quadruple) = self.img_group_transform(
                 img_group, self.img_scale,
                crop_history=data['img_meta']['crop_quadruple'],
                flip=data['img_meta']['flip'],
                keep_ratio=self.resize_keep_ratio,
                div_255=self.div_255,
                is_flow=True if modality == 'Flow' else False)

            if self.input_format == "NCTHW":
                # Convert [M x C x H x W] to [M' x C x T x H x W]
                # M = 1 * N_oversample * N_seg * L
                # M' = 1 * N_oversample * N_seg, T = L
                img_group = img_group.reshape(
                    (-1, self.num_segments,
                     self.new_length) + img_group.shape[1:])
                img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
                img_group = img_group.reshape((-1,) + img_group.shape[2:])

            else:
                data.update({
                    'img_group_{}'.format(i+1):
                    DC(to_tensor(img_group), stack=True, pad_dims=2),
                })

        return data
