import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (GroupImageTransform, BboxTransform)
from .utils import (to_tensor, random_scale)

_TIMESTAMP_BIAS = 600
_TIMESTAMP_START = 840  # 60*14min
_TIMESTAMP_END = 1860  # 60*31min
_FPS = 30


class RawFramesRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def video_id(self):
        return self._data[0]

    @property
    def timestamp(self):
        return int(self._data[1])

    @property
    def entity_box(self):
        x1 = float(self._data[2])
        y1 = float(self._data[3])
        x2 = float(self._data[4])
        y2 = float(self._data[5])
        return np.array([x1, y1, x2, y2])

    @property
    def label(self):
        return int(self._data[6])

    @property
    def entity_id(self):
        return int(self._data[7])


class AVADataset(Dataset):
    def __init__(self,
                 ann_file,
                 exclude_file,
                 label_file,
                 video_stat_file,
                 img_prefix,
                 img_norm_cfg,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=(340, 256),
                 input_size=None,
                 div_255=False,
                 size_divisor=None,
                 multiscale_mode='value',
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 with_label=False,
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

        # load annotations
        self.video_infos = self.load_annotations(
            ann_file, video_stat_file, exclude_file)
        self.ann_file = ann_file
        self.exclude_file = exclude_file
        self.label_file = label_file
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        # filter videos with no annotation during training
        if not test_mode:
            valid_inds = self._filter_records(exclude_file=exclude_file)
            print("{} out of {} frames are valid.".format(
                len(valid_inds), len(self.video_infos)))
            self.video_infos = [self.video_infos[i] for i in valid_inds]

        # normalization config
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals

        # parameters for frame fetching
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
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scales = img_scale
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
        self.div_255 = div_255

        # parameters for data augmentation
        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']
        # flip ratio
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        # with_label is False for RPN
        self.with_label = with_label

        # test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_group_transform = GroupImageTransform(
            size_divisor=None,
            crop_size=self.input_size,
            oversample=oversample,
            random_crop=random_crop, more_fix_crop=more_fix_crop,
            multiscale_crop=multiscale_crop, scales=scales,
            max_distort=max_distort,
            **self.img_norm_cfg)

        # input format
        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format

        self.bbox_transform = BboxTransform()

    def __len__(self):
        return len(self.video_infos)

    def arrange_annotations(self, records, video_stats):
        """ Rearrange the frame-level to annotation format similar to COCO
        [
            {
                'video_id': 'xxxxxxxxxxx',
                'timestamp': 902, ## in sec
                'width': 340,
                'height': 256,
                'fps': 30,
                'shot_info': (ss, tt),     ## in frame
                'ann': {
                    'bboxes': <np.ndarray> (n, 4),
                    'labels': <np.ndarray> (n, ), ==> <np.ndarray> (n, m=80)
                    'pids': <np.ndarray> (n, ),
                    'tracklets': <np.ndarray> (n, t, 4) (TODO: To be added)
                }
            },
            ...
        ]
        The 'ann' field is optional for testing
        """

        record_dict_by_image = dict()
        for record in records:
            image_key = "{},{:04d}".format(record.video_id, record.timestamp)
            if image_key not in record_dict_by_image:
                record_dict_by_image[image_key] = [record]
            else:
                record_dict_by_image[image_key].append(record)

        def merge(records):
            bboxes = []
            labels = []
            pids = []
            while len(records) > 0:
                r = records[0]
                rs = list(filter(lambda x: np.array_equal(
                    x.entity_box, r.entity_box), records))
                records = list(filter(lambda x: not np.array_equal(
                    x.entity_box, r.entity_box), records))
                bboxes.append(
                    r.entity_box * np.array([width, height, width, height]))
                valid_labels = np.stack([r.label for r in rs])
                padded_labels = np.pad(
                    valid_labels, (0, 81 - valid_labels.shape[0]),
                    'constant', constant_values=-1)
                labels.append(padded_labels)
                pids.append(r.entity_id)
            bboxes = np.stack(bboxes)
            labels = np.stack(labels)
            # print(bboxes)
            # print(labels)
            pids = np.stack(pids)
            return bboxes, labels, pids

        new_records = []
        for image_key in record_dict_by_image:
            video_id, timestamp = image_key.split(',')
            width = int(video_stats[video_id].split('x')[0])
            height = int(video_stats[video_id].split('x')[1])
            shot_info = (0, (_TIMESTAMP_END - _TIMESTAMP_START) * _FPS)

            bboxes, labels, pids = merge(record_dict_by_image[image_key])

            ann = dict(bboxes=bboxes,
                       labels=labels,
                       pids=pids)
            new_record = dict(video_id=video_id,
                              timestamp=int(timestamp),
                              width=width,
                              height=height,
                              shot_info=shot_info,
                              fps=_FPS,
                              ann=ann)
            new_records.append(new_record)

        return new_records

    def load_annotations(self, ann_file, video_stat_file, exclude_file=None):
        rawframe_records = [RawFramesRecord(
            x.strip().split(',')) for x in open(ann_file)]
        video_stats = [tuple(x.strip().split(' '))
                       for x in open(video_stat_file)]
        video_stats = {item[0]: item[1] for item in video_stats}
        return self.arrange_annotations(rawframe_records, video_stats)
        # return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.video_infos[idx]['ann']

    def _filter_records(self, exclude_file=None):
        valid_inds = []
        if exclude_file is not None:
            exclude_records = [x.strip().split(',')
                               for x in open(exclude_file)]
            for i, video_info in enumerate(self.video_infos):
                valid = True
                for vv, tt in exclude_records:
                    if (video_info['video_id'] == vv
                            and video_info['timestamp'] == int(tt)):
                        valid = False
                        break
                if valid:
                    valid_inds.append(i)
        else:
            for i, _ in enumerate(self.video_infos):
                valid_inds.append(i)
        return valid_inds

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

    def _load_image(self, directory, image_tmpl, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return [mmcv.imread(osp.join(directory, image_tmpl.format(idx)))]
        elif modality == 'Flow':
            x_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('x', idx)),
                flag='grayscale')
            y_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('y', idx)),
                flag='grayscale')
            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')

    def _get_frames(self, record, image_tmpl, modality, indice, skip_offsets):
        images = list()
        #
        p = indice - self.new_step
        for i, ind in enumerate(
                range(-2, -(self.old_length+1) // 2, -self.new_step)):
            seg_imgs = self._load_image(osp.join(
                self.img_prefix, record['video_id']),
                image_tmpl, modality, p + skip_offsets[i])
            images = seg_imgs + images
            if p - self.new_step >= record['shot_info'][0]:
                p -= self.new_step
        p = indice
        for i, ind in enumerate(
                range(0, (self.old_length+1) // 2, self.new_step)):
            seg_imgs = self._load_image(osp.join(
                self.img_prefix, record['video_id']),
                image_tmpl, modality, p + skip_offsets[i])
            images.extend(seg_imgs)
            if p + self.new_step < record['shot_info'][1]:
                p += self.new_step
        return images

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_imgs(idx)
        while True:
            data = self.prepare_train_imgs(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_imgs(self, idx):
        video_info = self.video_infos[idx]

        # load proposals if necessary
        if self.proposals is not None:
            image_key = "{},{:04d}".format(
                video_info['video_id'], video_info['timestamp'])
            if image_key not in self.proposals:
                return None
            proposals = self.proposals[image_key][: self.num_max_proposals]
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n,5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 4:
                proposals = proposals * np.array(
                    [video_info['width'], video_info['height'],
                     video_info['width'], video_info['height']])
            else:
                proposals = proposals * np.array(
                    [video_info['width'], video_info['height'],
                     video_info['width'], video_info['height'], 1.0])
            proposals = proposals.astype(np.float32)
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        # skip the record if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        gt_bboxes = gt_bboxes.astype(np.float32)

        indice = video_info['fps'] * \
            (video_info['timestamp'] - _TIMESTAMP_START) + 1
        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))))

        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        img_group = self._get_frames(
            video_info, image_tmpl, modality, indice, skip_offsets)

        # TODO: add extra augmentation ...

        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        (img_group, img_shape, pad_shape,
         scale_factor, crop_quadruple) = self.img_group_transform(
            img_group, img_scale,
            crop_history=None,
            flip=flip, keep_ratio=self.resize_keep_ratio,
            div_255=self.div_255,
            is_flow=True if modality == 'Flow' else False)
        ori_shape = (video_info['height'], video_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            crop_quadruple=crop_quadruple,
            flip=flip)

        # [L x C x H x W]
        if self.input_format == "NCTHW":
            img_group = np.transpose(img_group, (1, 0, 2, 3))
            # img_group = img_group[None, :]

        data.update(dict(
            img_group_0=DC(to_tensor(img_group), stack=True, pad_dims=2),
            img_meta=DC(img_meta, cpu_only=True)
        ))

        # handle the rest modalities using the same
        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            img_group = self._get_frames(
                video_info, image_tmpl, modality, indice, skip_offsets)

            # TODO: add extra augmentation ...

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            (img_group, img_shape, pad_shape,
             scale_factor, crop_quadruple) = self.img_group_transform(
                img_group, img_scale,
                crop_history=data['img_meta']['crop_quadruple'],
                flip=data['img_meta']['flip'],
                keep_ratio=self.resize_keep_ratio,
                div_255=self.div_255,
                is_flow=True if modality == 'Flow' else False)

            if self.input_format == "NCTHW":
                # Convert [L x C x H x W] to [C x L x H x W]
                img_group = np.transpose(img_group, (1, 0, 2, 3))
                # img_group = img_group[None, :]

            else:
                data.update({
                    'img_group_{}'.format(i+1): DC(
                        to_tensor(img_group), stack=True, pad_dims=2),
                })

        if self.proposals is not None:
            proposals = self.bbox_transform(
                proposals, img_shape, scale_factor, flip, crop=crop_quadruple)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
            data['proposals'] = DC(to_tensor(proposals))

        gt_bboxes = self.bbox_transform(
            gt_bboxes, img_shape, scale_factor, flip, crop=crop_quadruple)
        data['gt_bboxes'] = DC(to_tensor(gt_bboxes))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))

        return data

    def prepare_test_imgs(self, idx):
        video_info = self.video_infos[idx]

        # load proposals if necessary
        if self.proposals is not None:
            image_key = "{},{:04d}".format(
                video_info['video_id'], video_info['timestamp'])
            if image_key not in self.proposals:
                proposal = np.array([[0, 0, 1, 1, 1]], dtype=float)
            else:
                proposal = self.proposals[image_key][: self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n,5), '
                    'but found {}'.format(proposal.shape))
            if proposal.shape[1] == 4:
                proposal = proposal * np.array(
                    [video_info['width'], video_info['height'],
                     video_info['width'], video_info['height']])
            else:
                proposal = proposal * np.array(
                    [video_info['width'], video_info['height'],
                     video_info['width'], video_info['height'], 1.0])
            proposal = proposal.astype(np.float32)
        else:
            proposal = None

        def prepare_single(img_group, scale,
                           crop_quadruple, flip, proposal=None):
            (_img_group, img_shape, pad_shape,
             scale_factor, crop_quadruple) = self.img_group_transform(
                img_group, scale,
                crop_history=crop_quadruple,
                 flip=flip,
                 keep_ratio=self.resize_keep_ratio)
            _img_group = to_tensor(_img_group)
            _img_meta = dict(
                ori_shape=(video_info['height'], video_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                crop_quadruple=crop_quadruple,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(
                    proposal, img_shape,
                    scale_factor, flip, crop=crop_quadruple)
                _proposal = np.hstack(
                    [_proposal, score] if score is not None else _proposal)
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img_group, _img_meta, _proposal

        indice = video_info['fps'] * \
            (video_info['timestamp'] - _TIMESTAMP_START) + 1
        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))))

        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities, self.image_tmpls)):
            img_group = self._get_frames(
                video_info, image_tmpl, modality, indice, skip_offsets)
            img_groups = []
            img_metas = []
            proposals = []
            for scale in self.img_scales:
                _img_group, _img_meta, _proposal = prepare_single(
                    img_group, scale, None, False, proposal)
                if self.input_format == "NCTHW":
                    # Convert [L x C x H x W] to [C x L x H x W]
                    _img_group = np.transpose(_img_group, (1, 0, 2, 3))
                img_groups.append(_img_group)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
                if self.flip_ratio > 0:
                    _img_group, _img_meta, _proposal = prepare_single(
                        img_group, scale, None, True, proposal)
                    if self.input_format == "NCTHW":
                        # Convert [L x C x H x W] to [C x L x H x W]
                        _img_group = np.transpose(_img_group, (1, 0, 2, 3))
                    img_groups.append(_img_group)
                    img_metas.append(DC(_img_meta, cpu_only=True))
                    proposals.append(_proposal)
            data['img_group_{}'.format(i)] = img_groups
            if i == 0:
                data['img_meta'] = img_metas
            if self.proposals is not None:
                data['proposals'] = proposals

        return data
