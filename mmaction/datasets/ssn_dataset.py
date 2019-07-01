import mmcv
import numpy as np
import math
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (GroupImageTransform)
from .utils import (to_tensor, parse_directory,
                    process_localize_proposal_list,
                    load_localize_proposal_file)

from mmaction.core.bbox1d import temporal_iou


class SSNInstance(object):
    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

        self.loc_reg = None
        self.size_reg = None

    def compute_regression_targets(self, gt_list, fg_thresh):
        if self.best_iou < fg_thresh:
            # background proposals do not need this
            return

        # find the groundtruth instance with the highest IOU
        ious = [temporal_iou((self.start_frame, self.end_frame),
                             (gt.start_frame, gt.end_frame)) for gt in gt_list]
        best_gt_id = np.argmax(ious)

        best_gt = gt_list[best_gt_id]

        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2

        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift proportional to the proposal duration
        # (2). logairthm of the groundtruth duration over proposal duration

        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except ValueError:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise ValueError("gt_size / prop_size should be valid.")

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

    @property
    def regression_targets(self):
        target = ([self.loc_reg, self.size_reg] if self.loc_reg is not None
                  else [0., 0.])
        return target


class SSNVideoRecord(object):
    def __init__(self, prop_record):
        self._data = prop_record

        frame_count = int(self._data[1])

        self.gt = [
            SSNInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]),
                        best_iou=1.0)
            for x in self._data[2] if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            SSNInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                        best_iou=float(x[1]), overlap_self=float(x[2]))
            for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(
            filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def video_id(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True):
        fg = [p for p in self.proposals if p.best_iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)

        for x in fg:
            x.compute_regression_targets(self.gt, fg_thresh)

        return fg

    def get_negatives(self, incomplete_iou_thresh, bg_iou_thresh,
                      bg_coverage_thresh=0.01, incomplete_overlap_thresh=0.7):
        tag = [0] * len(self.proposals)

        incomplete_props = []
        background_props = []

        for i in range(len(tag)):
            if self.proposals[i].best_iou < incomplete_iou_thresh and \
                    self.proposals[i].overlap_self > incomplete_overlap_thresh:
                tag[i] = 1  # incomplete
                incomplete_props.append(self.proposals[i])

        for i in range(len(tag)):
            if tag[i] == 0 and \
                    self.proposals[i].best_iou < bg_iou_thresh and \
                    self.proposals[i].coverage > bg_coverage_thresh:
                background_props.append(self.proposals[i])
        return incomplete_props, background_props


class SSNDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 train_cfg,
                 test_cfg,
                 video_centric=True,
                 reg_stats=None,
                 body_seg=5,
                 aug_seg=2,
                 aug_ratio=(0.5, 0.5),
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 input_size=None,
                 div_255=False,
                 size_divisor=None,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 filter_gt=True,
                 test_mode=False,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW',
                 verbose=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # normalization config
        self.img_norm_cfg = img_norm_cfg

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.verbose = verbose

        # load annotations
        if 'normalized_' in ann_file:
            self.proposal_file = ann_file.replace('normalized_', '')
            if not osp.exists(self.proposal_file):
                print('{} does not exist. Converting from {}'.format(
                    self.proposal_file, ann_file))
                frame_dict = parse_directory(
                    self.img_prefix, key_func=lambda x: x.split('/')[-1])
                process_localize_proposal_list(
                    ann_file, self.proposal_file, frame_dict)
                print('Finished conversion.')
        else:
            self.proposal_file = ann_file

        proposal_infos = load_localize_proposal_file(self.proposal_file)
        self.video_infos = [SSNVideoRecord(p) for p in proposal_infos]

        # filter videos with no annotation during training
        if filter_gt or not test_mode:
            valid_inds = self._filter_records()
        print("{} out of {} videos are valid.".format(
            len(valid_inds), len(self.video_infos)))
        self.video_infos = [self.video_infos[i] for i in valid_inds]

        self.video_dict = {
            record.video_id: record for record in self.video_infos}

        # construct three pools:
        # 1. Foreground
        # 2. Background
        # 3. Incomplete
        self.fg_pool = []
        self.bg_pool = []
        self.incomp_pool = []

        for v in self.video_infos:
            self.fg_pool.extend([(v.video_id, prop)
                                 for prop in v.get_fg(
                                     self.train_cfg.ssn.assigner.fg_iou_thr,
                                     self.train_cfg.ssn.sampler.
                                     add_gt_as_proposals)])
            incomp, bg = v.get_negatives(
                self.train_cfg.ssn.assigner.incomplete_iou_thr,
                self.train_cfg.ssn.assigner.bg_iou_thr,
                self.train_cfg.ssn.assigner.bg_coverage_thr,
                self.train_cfg.ssn.assigner.incomplete_overlap_thr)
            self.incomp_pool.extend([(v.video_id, prop) for prop in incomp])
            self.bg_pool.extend([v.video_id, prop] for prop in bg)

        if reg_stats is None:
            self.reg_stats = self._compute_regression_stats()
        else:
            self.reg_stats = reg_stats

        self.video_centric = video_centric

        self.body_seg = body_seg
        self.aug_seg = aug_seg

        if isinstance(aug_ratio, (int, float)):
            self.aug_ratio = (aug_ratio, aug_ratio)
        else:
            assert isinstance(aug_ratio, (tuple, list))
            assert len(aug_ratio) == 2
            self.aug_ratio = aug_ratio

        denum = self.train_cfg.ssn.sampler.fg_ratio + \
            self.train_cfg.ssn.sampler.bg_ratio + \
            self.train_cfg.ssn.sampler.incomplete_ratio
        self.fg_per_video = int(self.train_cfg.ssn.sampler.num_per_video *
                                (self.train_cfg.ssn.sampler.fg_ratio / denum))
        self.bg_per_video = int(self.train_cfg.ssn.sampler.num_per_video *
                                (self.train_cfg.ssn.sampler.bg_ratio / denum))
        self.incomplete_per_video = self.train_cfg.ssn.sampler.num_per_video -\
            self.fg_per_video - self.bg_per_video

        self.test_interval = self.test_cfg.ssn.sampler.test_interval

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
        self.img_scale = img_scale
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
        self.div_255 = div_255

        # flip ratio
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        # test mode or not
        self.filter_gt = filter_gt
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        assert oversample in [None, 'three_crop', 'ten_crop']
        self.oversample = oversample
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

        if self.verbose:
            print("""
            SSNDataset: proposal file {prop_file} parsed.

            There are {pnum} usable proposals from {vnum} videos.
            {fnum} foreground proposals
            {inum} incomplete proposals
            {bnum} background proposals

            Sample config:
            FG/BG/INC: {fr}/{br}/{ir}
            Video Centric: {vc}

            Regression Stats:
            Location: mean {stats[0][0]:.05f} std {stats[1][0]:.05f}
            Duration: mean {stats[0][1]:.05f} std {stats[1][1]:.05f}
            """.format(prop_file=self.proposal_file,
                       pnum=len(self.fg_pool) + len(self.bg_pool) +
                       len(self.incomp_pool),
                       fnum=len(self.fg_pool), inum=len(self.incomp_pool),
                       bnum=len(self.bg_pool),
                       fr=self.fg_per_video, br=self.bg_per_video,
                       ir=self.incomplete_per_video,
                       vnum=len(self.video_infos), vc=self.video_centric,
                       stats=self.reg_stats))
        else:
            print("""
            SSNDataset: proposal file {prop_file} parsed.
            """.format(prop_file=self.proposal_file))

    def __len__(self):
        return len(self.video_infos)

    def get_ann_info(self, idx):
        return self.video_infos[idx]

    def get_all_gt(self):
        gt_list = []
        for video in self.video_infos:
            vid = video.video_id
            gt_list.extend(
                [[vid, x.label - 1, x.start_frame / video.num_frames,
                  x.end_frame / video.num_frames] for x in video.gt])
        return gt_list

    def _filter_records(self):
        valid_inds = []
        for i, x in enumerate(self.video_infos):
            if len(x.gt) > 0:
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

    def _video_centric_sampling(self, record):
        fg = record.get_fg(self.train_cfg.ssn.assigner.fg_iou_thr,
                           self.train_cfg.ssn.sampler.add_gt_as_proposals)
        incomp, bg = record.get_negatives(
            self.train_cfg.ssn.assigner.incomplete_iou_thr,
            self.train_cfg.ssn.assigner.bg_iou_thr,
            self.train_cfg.ssn.assigner.bg_coverage_thr,
            self.train_cfg.ssn.assigner.incomplete_overlap_thr)

        def sample_video_proposals(proposal_type, video_id, video_pool,
                                   requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is nothing in the video pool,
                # go fetch from the dataset pool
                return [(dataset_pool[x], proposal_type)
                        for x in np.random.choice(
                        len(dataset_pool), requested_num, replace=False)]
            else:
                replicate = len(video_pool) < requested_num
                idx = np.random.choice(
                    len(video_pool), requested_num, replace=replicate)
                return [((video_id, video_pool[x]), proposal_type)
                        for x in idx]

        out_props = []
        out_props.extend(sample_video_proposals(
            0, record.video_id, fg, self.fg_per_video, self.fg_pool))
        out_props.extend(sample_video_proposals(
            1, record.video_id, incomp,
            self.incomplete_per_video, self.incomp_pool))
        out_props.extend(sample_video_proposals(
            2, record.video_id, bg, self.bg_per_video, self.bg_pool))

        return out_props

    def _random_sampling(self):
        out_props = []

        out_props.extend([(x, 0) for x in np.random.choice(
            self.fg_pool, self.fg_per_video, replace=False)])
        out_props.extend([(x, 1) for x in np.random.choice(
            self.incomp_pool, self.incomplete_per_video, replace=False)])
        out_props.extend([(x, 2) for x in np.random.choice(
            self.bg_pool, self.bg_per_video, replace=False)])

        return out_props

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_imgs(idx)
        else:
            return self.prepare_train_imgs(idx)

    def _compute_regression_stats(self):
        if self.verbose:
            print('Computing regression target normlizing constants')
        targets = []
        for video in self.video_infos:
            fg = video.get_fg(self.train_cfg.ssn.assigner.fg_iou_thr, False)
            for p in fg:
                targets.append(list(p.regression_targets))

        return np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))

    def _sample_indices(self, valid_length, num_seg):
        average_duration = (valid_length + 1) // num_seg
        if average_duration > 0:
            offsets = np.multiply(list(range(num_seg)), average_duration) + \
                np.random.randint(average_duration, size=num_seg)
        elif valid_length > num_seg:
            offsets = np.sort(np.random.randint(valid_length, size=num_seg))
        else:
            offsets = np.zeros((num_seg, ))

        return offsets

    def _get_val_indices(self, valid_length, num_seg):
        if valid_length > num_seg:
            tick = valid_length / float(num_seg)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(num_seg)])
        else:
            offsets = np.zeros((num_seg, ))

        return offsets

    def _sample_ssn_indices(self, prop, frame_cnt):
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame

        duration = end_frame - start_frame + 1
        assert duration != 0
        valid_length = duration - self.old_length

        valid_starting = max(
            1, start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(frame_cnt - self.old_length + 1,
                           end_frame + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - \
            self.old_length + 1
        valid_ending_length = valid_ending - end_frame - self.old_length + 1

        starting_scale = (valid_starting_length +
                          self.old_length + 1) / (duration * self.aug_ratio[0])
        ending_scale = (valid_ending_length + self.old_length +
                        1) / (duration * self.aug_ratio[1])

        starting_offsets = self._sample_indices(
            valid_starting_length, self.aug_seg[0]) if self.random_shift \
            else self._get_val_indices(valid_starting_length, self.aug_seg[0])
        starting_offsets += valid_starting
        course_offsets = self._sample_indices(
            valid_length, self.body_seg) if self.random_shift \
            else self._get_val_indices(valid_length, self.body_seg)
        course_offsets += start_frame
        ending_offsets = self._sample_indices(
            valid_ending_length, self.aug_seg[1]) if self.random_shift \
            else self._get_val_indices(valid_ending_length, self.aug_seg[1])
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        stage_split = [self.aug_seg[0], self.aug_seg[0] + self.body_seg,
                       self.aug_seg[0] + self.body_seg + self.aug_seg[1]]
        return offsets, starting_scale, ending_scale, stage_split

    def prepare_train_imgs(self, idx):
        if self.video_centric:
            video_info = self.video_infos[idx]
            props = self._video_centric_sampling(video_info)
        else:
            props = self._random_sampling()

        out_frames = []
        for _ in range(len(self.modalities)):
            out_frames.append([])
        out_prop_scaling = []
        out_prop_type = []
        out_prop_labels = []
        out_prop_reg_targets = []
        out_img_meta = []

        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))))

        for idx, prop in enumerate(props):
            frame_cnt = self.video_dict[prop[0][0]].num_frames

            (prop_indices, starting_scale, ending_scale,
             stage_split) = self._sample_ssn_indices(prop[0][1], frame_cnt)

            modality = self.modalities[0]
            image_tmpl = self.image_tmpls[0]

            # handle the first modality
            img_group = []
            for idx, seg_ind in enumerate(prop_indices):
                for i, x in enumerate(
                        range(0, self.old_length, self.new_step)):
                    img_group.extend(self._load_image(
                        osp.join(self.img_prefix, prop[0][0]),
                        image_tmpl, modality,
                        min(frame_cnt, int(seg_ind) + x + skip_offsets[i])))

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
            out_img_meta.append(img_meta)
            # [L x C x H x W]
            if self.input_format == "NCTHW":
                img_group = np.transpose(img_group, (1, 0, 2, 3))
            out_frames[0].append(img_group)

            for i, (modality, image_tmpl) in enumerate(
                    zip(self.modalities[1:], self.image_tmpls[1:])):
                img_group = []
                for idx, seg_ind in enumerate(prop_indices):
                    for x in range(0, self.old_length, self.new_step):
                        img_group.extend(self._load_image(osp.join(
                            self.img_prefix, prop[0][0]), image_tmpl, modality,
                            int(seg_ind) + x + skip_offsets[i]))

                flip = True if np.random.rand() < self.flip_ratio else False
                (img_group, img_shape, pad_shape,
                 scale_factor, crop_quadruple) = self.img_group_transform(
                    img_group, self.img_scale,
                    crop_history=img_meta['crop_quadruple'],
                    flip=img_meta['flip'], keep_ratio=self.resize_keep_ratio,
                    div_255=self.div_255,
                    is_flow=True if modality == 'Flow' else False)
                # [L x C x H x W]
                if self.input_format == "NCTHW":
                    img_group = np.transpose(img_group, (1, 0, 2, 3))
                out_frames[i + 1].append(img_group)

            if prop[1] == 0:
                label = prop[0][1].label
            elif prop[1] == 1:
                label = prop[0][1].label
            elif prop[1] == 2:
                label = 0
            else:
                raise ValueError("proposal type should be 0, 1, or 2")
            out_prop_scaling.append([starting_scale, ending_scale])
            out_prop_labels.append(label)
            out_prop_type.append(prop[1])

            if prop[1] == 0:
                reg_targets = prop[0][1].regression_targets
                reg_targets = ((reg_targets[0] - self.reg_stats[0][0]) / \
                    self.reg_stats[1][0],
                (reg_targets[1] - self.reg_stats[0][1]) / \
                    self.reg_stats[1][1])
            else:
                reg_targets = (0., 0.)

            out_prop_reg_targets.append(reg_targets)

        for i in range(len(out_frames)):
            out_frames[i] = np.array(out_frames[i])

        data.update({
            'img_group_0': DC(to_tensor(out_frames[0]), stack=True,
                              pad_dims=2),
            'img_meta': DC(out_img_meta, cpu_only=True)
        })

        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            data.update({
                'img_group_{}'.format(i+1): DC(to_tensor(out_frames[i+1]),
                                               stack=True, pad_dims=2)
            })

        data['reg_targets'] = DC(to_tensor(
            np.array(out_prop_reg_targets, dtype=np.float32)),
            stack=True, pad_dims=None)
        data['prop_scaling'] = DC(to_tensor(
            np.array(out_prop_scaling, dtype=np.float32)),
            stack=True, pad_dims=None)
        data['prop_labels'] = DC(
            to_tensor(np.array(out_prop_labels)), stack=True, pad_dims=None)
        data['prop_type'] = DC(
            to_tensor(np.array(out_prop_type)), stack=True, pad_dims=None)

        return data

    def prepare_test_imgs(self, idx):
        video_info = self.video_infos[idx]

        props = video_info.proposals
        video_id = video_info.video_id
        frame_cnt = video_info.num_frames
        frame_ticks = np.arange(
            0, frame_cnt - self.old_length, self.test_interval, dtype=int) + 1

        num_sampled_frames = len(frame_ticks)

        if len(props) == 0:
            props.append(SSNInstance(0, frame_cnt - 1, frame_cnt))

        rel_prop_list = []
        proposal_tick_list = []
        scaling_list = []
        out_frames = []
        for _ in range(len(self.modalities)):
            out_frames.append([])
        out_img_meta = []
        for proposal in props:
            rel_prop = (proposal.start_frame / frame_cnt,
                        proposal.end_frame / frame_cnt)
            rel_duration = rel_prop[1] - rel_prop[0]
            rel_starting_duration = rel_duration * self.aug_ratio[0]
            rel_ending_duration = rel_duration * self.aug_ratio[1]
            rel_starting = rel_prop[0] - rel_starting_duration
            rel_ending = rel_prop[1] + rel_ending_duration

            real_rel_starting = max(0.0, rel_starting)
            real_rel_ending = min(1.0, rel_ending)

            starting_scaling = (
                rel_prop[0] - real_rel_starting) / rel_starting_duration
            ending_scaling = (real_rel_ending -
                              rel_prop[1]) / rel_ending_duration

            proposal_ticks = (int(real_rel_starting * num_sampled_frames),
                              int(rel_prop[0] * num_sampled_frames),
                              int(rel_prop[1] * num_sampled_frames),
                              int(real_rel_ending * num_sampled_frames))

            rel_prop_list.append(rel_prop)
            proposal_tick_list.append(proposal_ticks)
            scaling_list.append((starting_scaling, ending_scaling))

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))))

        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]

        skip_offsets = np.random.randint(
            self.new_step, size=self.old_length // self.new_step)

        # handle the first modality
        img_group = []
        for idx, seg_ind in enumerate(frame_ticks):
            for x in range(0, self.old_length, self.new_step):
                img_group.extend(self._load_image(
                    osp.join(self.img_prefix, video_id), image_tmpl,
                    modality, int(seg_ind) + x))

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
        out_img_meta.append(img_meta)
        # [L x C x H x W]
        if self.input_format == "NCTHW":
            img_group = np.transpose(img_group, (1, 0, 2, 3))
        out_frames[0].append(img_group)

        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            img_group = []
            for idx, seg_ind in enumerate(frame_ticks):
                for j, x in enumerate(
                        range(0, self.old_length, self.new_step)):
                    img_group.extend(self._load_image(osp.join(
                        self.img_prefix, video_id), image_tmpl, modality,
                        int(seg_ind) + x + skip_offsets[j]))

            flip = True if np.random.rand() < self.flip_ratio else False
            (img_group, img_shape, pad_shape,
             scale_factor, crop_quadruple) = self.img_group_transform(
                img_group, self.img_scale,
                crop_history=img_meta['crop_quadruple'],
                flip=img_meta['flip'], keep_ratio=self.resize_keep_ratio,
                div_255=self.div_255,
                is_flow=True if modality == 'Flow' else False)
            # [L x C x H x W]
            if self.input_format == "NCTHW":
                img_group = np.transpose(img_group, (1, 0, 2, 3))
            out_frames[i + 1].append(img_group)

        for i in range(len(out_frames)):
            if self.oversample == 'ten_crop':
                num_crop = 10
            elif self.oversample == 'three_crop':
                num_crop = 3
            else:
                num_crop = 1
            out_frames[i] = np.array(out_frames[i])
            out_frames[i] = out_frames[i].reshape(
                (num_crop, -1) + out_frames[i].shape[2:])

        data.update({
            'img_group_0': DC(to_tensor(out_frames[0]), cpu_only=True),
            'img_meta': DC(out_img_meta, cpu_only=True)
        })

        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            data.update({
                'img_group_{}'.format(i+1):
                DC(to_tensor(out_frames[i+1]), cpu_only=True)
            })

        data['rel_prop_list'] = DC(to_tensor(
            np.array(rel_prop_list, dtype=np.float32)),
            stack=True, pad_dims=None)
        data['scaling_list'] = DC(
            to_tensor(np.array(scaling_list, dtype=np.float32)),
            stack=True, pad_dims=None)
        data['prop_tick_list'] = DC(
            to_tensor(np.array(proposal_tick_list)),
            stack=True, pad_dims=None)
        data['reg_stats'] = DC(to_tensor(self.reg_stats),
                               stack=True, pad_dims=None)

        return data
