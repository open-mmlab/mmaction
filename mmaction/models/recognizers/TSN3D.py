from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS

import torch


@RECOGNIZERS.register_module
class TSN3D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 flownet=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 fcn_testing=False):

        super(TSN3D, self).__init__()
        self.fcn_testing = fcn_testing
        self.backbone = builder.build_backbone(backbone)

        if flownet is not None:
            self.flownet = builder.build_flownet(flownet)

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            raise NotImplementedError

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            raise NotImplementedError

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            raise NotImplementedError

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    @property
    def with_flownet(self):
        return hasattr(self, 'flownet') and self.flownet is not None

    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(TSN3D, self).init_weights()
        self.backbone.init_weights()

        if self.with_flownet:
            self.flownet.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat_with_flow(self, img_group,
                               trajectory_forward=None,
                               trajectory_backward=None):
        x = self.backbone(img_group,
                          trajectory_forward=trajectory_forward,
                          trajectory_backward=trajectory_backward)
        return x

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape((-1, ) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs

        if self.with_flownet:
            if self.flownet.multiframe:
                img_forward = img_group[:, :, 1:, :, :]
                if self.flownet.flip_rgb:
                    img_forward = img_forward.flip(1)
                img_forward = img_forward.transpose(1, 2).contiguous().view(
                    (img_forward.size(0), -1,
                     img_forward.size(3), img_forward.size(4)))
                trajectory_forward, photometric_forward, ssim_forward, smooth_forward = self.flownet(
                    img_forward)
                img_backward = img_group.flip(2)[:, :, 1:, :, :]
                if self.flownet.rgb_disorder:
                    img_backward = img_backward.flip(1)
                img_backward = img_backward.transpose(1, 2).contiguous().view(
                    (img_backward.size(0), -1,
                     img_backward.size(3), img_backward.size(4)))
                trajectory_backward, photometric_backward, ssim_backward, smooth_backward = self.flownet(
                    img_backward)
            else:
                # TODO: Wrap it into a function, e.g. ImFlows2ImFlowStack
                num_frames = img_group.size(2)
                traj_forwards, traj_backwards = [], []
                photometric_forwards, photometric_backwards = [], []
                ssim_forwards, ssim_backwards = [], []
                smooth_forwards, smooth_backwards = [], []
                for i in range(1, num_frames - 1):
                    img_forward = img_group[:, :, i:i+2, :, :]
                    if self.flownet.flip_rgb:
                        img_forward = img_forward.flip(1)
                    img_forward = img_forward.transpose(1, 2).contiguous().view(
                        (img_forward.size(0), -1,
                         img_forward.size(3), img_forward.size(4)))
                    traj_forward, photometric_forward, ssim_forward, smooth_forward = self.flownet(
                        img_forward)
                    traj_forwards.append(traj_forward)
                    photometric_forwards.append(photometric_forward)
                    ssim_forwards.append(ssim_forward)
                    smooth_forwards.append(smooth_forward)
                    img_backward = img_group[
                        :, :,
                        num_frames - i - 1: num_frames - i + 1, :, :].flip(2)
                    if self.flownet.flip_rgb:
                        img_backward = img_backward.flip(1)
                    img_backward = img_backward.transpose(1, 2).contiguous().view(
                        (img_backward.size(0), -1,
                         img_backward.size(3), img_backward.size(4)))
                    traj_backward, photometric_backward, ssim_backward, smooth_backward = self.flownet(
                        img_backward)
                    traj_backwards.append(traj_backward)
                    photometric_backwards.append(photometric_backward)
                    ssim_backwards.append(ssim_backward)
                    smooth_backwards.append(smooth_backward)

                def _organize_trajectories(trajectory_lvls_pairs):
                    res = [[]] * len(trajectory_lvls_pairs[0])
                    for trajectory_lvls in trajectory_lvls_pairs:
                        for i, trajectory in enumerate(trajectory_lvls):
                            res[i].append(trajectory)
                    for i in range(len(trajectory_lvls_pairs[0])):
                        res[i] = torch.cat(res[i], 1)
                    return tuple(res)

                def _organize_loss_outs(loss_outs_lvls_pairs):
                    L = len(loss_outs_lvls_pairs)
                    num_level = len(loss_outs_lvls_pairs[0])
                    num_item = len(loss_outs_lvls_pairs[0][0])
                    res = []
                    for i in range(num_level):
                        res_level = []
                        for j in range(num_item):
                            outs = []
                            for k in range(L):
                                outs.append(loss_outs_lvls_pairs[k][i][j])
                            res_level.append(outs)
                        res.append(res_level)
                    for i in range(num_level):
                        for j in range(num_item):
                            res[i][j] = torch.cat(res[i][j], 1)
                        res[i] = tuple(res[i])
                    return tuple(res)

                trajectory_forward = _organize_trajectories(traj_forwards)
                trajectory_backward = _organize_trajectories(traj_backwards)
                photometric_forward = _organize_loss_outs(photometric_forwards)
                photometric_backward = _organize_loss_outs(
                    photometric_backwards)
                ssim_forward = _organize_loss_outs(ssim_forwards)
                ssim_backward = _organize_loss_outs(ssim_backwards)
                smooth_forward = _organize_loss_outs(smooth_forwards)
                smooth_backward = _organize_loss_outs(smooth_backwards)

            x = self.extract_feat_with_flow(
                img_group[:, :, 1:-1, :, :],
                trajectory_forward=trajectory_forward,
                trajectory_backward=trajectory_backward)
        else:
            x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = x.reshape((-1, num_seg) + x.shape[1:])
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        losses = dict()
        if self.with_flownet:
            losses.update(self.flownet.loss(photometric_forward,
                                            ssim_forward, smooth_forward,
                                            direction='forward'))
            losses.update(self.flownet.loss(photometric_backward,
                                            ssim_backward, smooth_backward,
                                            direction='backward'))
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = gt_label.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape((-1, ) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs

        if self.with_flownet:
            if self.flownet.multiframe:
                img_forward = img_group[:, :, 1:, :, :]
                if self.flownet.flip_rgb:
                    img_forward = img_forward.flip(1)
                img_forward = img_forward.transpose(1, 2).contiguous().view(
                    (img_forward.size(0), -1,
                     img_forward.size(3), img_forward.size(4)))
                trajectory_forward, _, _, _ = self.flownet(
                    img_forward, train=False)
                img_backward = img_group.flip(2)[:, :, 1:, :, :]
                if self.flownet.flip_rgb:
                    img_backward = img_backward.flip(1)
                img_backward = img_backward.transpose(1, 2).contiguous().view(
                    (img_backward.size(0), -1,
                     img_backward.size(3), img_backward.size(4)))
                trajectory_backward, _, _, _ = self.flownet(
                    img_backward, train=False)
            else:
                # TODO: Wrap it into a function, e.g. ImFlows2ImFlowStack
                num_frames = img_group.size(2)
                traj_forwards, traj_backwards = [], []
                for i in range(1, num_frames - 1):
                    img_forward = img_group[:, :, i:i+2, :, :]
                    if self.flownet.rgb_disorder:
                        img_forward = img_forward.flip(1)
                    img_forward = img_forward.transpose(1, 2).contiguous().view(
                        (img_forward.size(0), -1,
                         img_forward.size(3), img_forward.size(4)))
                    traj_forward, _, _, _ = self.flownet(
                        img_forward, train=False)
                    traj_forwards.append(traj_forward)
                    img_backward = img_group[
                        :, :, num_frames - i - 1:
                        num_frames - i + 1, :, :].flip(2)
                    if self.flownet.rgb_disorder:
                        img_backward = img_backward.flip(1)
                    img_backward = img_backward.transpose(1, 2).contiguous().view(
                        (img_backward.size(0), -1,
                         img_backward.size(3), img_backward.size(4)))
                    traj_backward, _, _, _ = self.flownet(
                        img_backward, train=False)
                    traj_backwards.append(traj_backward)

                def _organize_trajectories(trajectory_lvls_pairs):
                    res = [[]] * len(trajectory_lvls_pairs[0])
                    for trajectory_lvls in trajectory_lvls_pairs:
                        for i, trajectory in enumerate(trajectory_lvls):
                            res[i].append(trajectory)
                    for i in range(len(trajectory_lvls_pairs[0])):
                        res[i] = torch.cat(res[i], 1)
                    return tuple(res)

                trajectory_forward = _organize_trajectories(traj_forwards)
                trajectory_backward = _organize_trajectories(traj_backwards)

            x = self.extract_feat_with_flow(
                img_group[:, :, 1:-1, :, :],
                trajectory_forward=trajectory_forward,
                trajectory_backward=trajectory_backward)
        else:
            x = self.extract_feat(img_group)

        if self.fcn_testing:
            x = self.cls_head(x)

            prob1 = torch.nn.functional.softmax(x.mean([2,3,4]), 1).mean(0, keepdim=True)
            return prob1.detach().cpu().numpy()


        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = x.reshape((-1, num_seg) + x.shape[1:])
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy()
