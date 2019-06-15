import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin
from .. import builder
from ..registry import DETECTORS
from mmaction.core.bbox2d import (bbox2roi, bbox2result,
                                  build_assigner, build_sampler)


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 dropout_ratio=0,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self):
        super(TwoStageDetector, self).init_weights()
        self.backbone.init_weights()
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, image_group):
        x = self.backbone(image_group)
        if self.with_neck:
            x = self.neck()
        else:
            if not isinstance(x, (list, tuple)):
                x = (x, )
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        x = self.extract_feat(img_group)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            x_slice = (xx[:, :, xx.size(2) // 2, :, :] for xx in x)
            rpn_outs = self.rpn_head(x_slice)
            rpn_loss_inputs = rpn_outs + \
                (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if not self.train_cfg.train_detector:
            proposal_list = []
            for proposal in proposals:
                select_inds = proposal[:, 4] >= min(
                    self.train_cfg.person_det_score_thr,
                    max(proposal[:, 4]))
                proposal_list.append(proposal[select_inds])

        # assign gts and sample proposals
        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img_group.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            if self.dropout is not None:
                bbox_feats = self.dropout(bbox_feats)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            if not self.train_cfg.train_detector:
                loss_bbox.pop('loss_person_cls')
            losses.update(loss_bbox)

        return losses

    def simple_test(self, num_modalities, img_meta,
                    proposals=None, rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        assert num_modalities == 1
        img_group = kwargs['img_group_0'][0]
        x = self.extract_feat(img_group)

        if proposals is None:
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn)
        else:
            proposal_list = []
            for proposal in proposals:
                proposal = proposal[0, ...]
                if not self.test_cfg.train_detector:
                    select_inds = proposal[:, 4] >= min(
                        self.test_cfg.person_det_score_thr,
                        max(proposal[:, 4]))
                    proposal = proposal[select_inds]
                proposal_list.append(proposal)

        img_meta = img_meta[0]

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes,
                                   thr=self.test_cfg.rcnn.action_thr)

        return bbox_results

    def aug_test(self, num_modalities, img_metas,
                 proposals=None, rescale=False,
                 **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes will fit the scale
        of imgs[0]
        """
        assert num_modalities == 1
        img_groups = kwargs['img_group_0']
        if proposals is None:
            proposal_list = self.aug_test_rpn(
                self.extract_feats(img_groups), img_metas, self.test_cfg.rpn)
        else:
            # TODO: need check
            proposal_list = []
            for proposal in proposals:
                proposal = proposal[0, ...]
                if not self.test_cfg.train_detector:
                    select_inds = proposal[:, 4] >= min(
                        self.test_cfg.person_det_score_thr,
                        max(proposal[:, 4]))
                    proposal = proposal[select_inds]
                proposal_list.append(proposal)

        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(img_groups), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes,
                                   thr=self.test_cfg.rcnn.action_thr)

        return bbox_results
