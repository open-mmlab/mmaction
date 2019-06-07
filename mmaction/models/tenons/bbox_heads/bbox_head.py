import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.core.bbox2d import delta2bbox, bbox_target
from mmaction.core.post_processing import multiclass_nms, singleclass_nms
from mmaction.losses import (weighted_cross_entropy, weighted_smoothl1,
                             multilabel_accuracy,
                             weighted_binary_cross_entropy,
                             weighted_multilabel_binary_cross_entropy)
from ...registry import HEADS


@HEADS.register_module
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_temporal_pool=False,
                 with_spatial_pool=False,
                 temporal_pool_type='avg',
                 spatial_pool_type='max',
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 multilabel_classification=True,
                 reg_class_agnostic=True,
                 nms_class_agnostic=True):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_temporal_pool = with_temporal_pool
        self.with_spatial_pool = with_spatial_pool
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.multilabel_classification = multilabel_classification
        self.reg_class_agnostic = reg_class_agnostic
        self.nms_class_agnostic = nms_class_agnostic

        in_channels = self.in_channels
        if self.with_temporal_pool:
            if self.temporal_pool_type == 'avg':
                self.temporal_pool = nn.AvgPool3d((roi_feat_size[0], 1, 1))
            else:
                self.temporal_pool = nn.MaxPool3d((roi_feat_size[0], 1, 1))
        if self.with_spatial_pool:
            if self.spatial_pool_type == 'avg':
                self.spatial_pool = nn.AvgPool3d((1, roi_feat_size[1], roi_feat_size[2]))
            else:
                self.spatial_pool = nn.MaxPool3d((1, roi_feat_size[1], roi_feat_size[2]))
        if not self.with_temporal_pool and not self.with_spatial_pool:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_temporal_pool:
            x = self.temporal_pool(x)
        if self.with_spatial_pool:
            x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             class_weights,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            if not self.multilabel_classification:
                assert len(labels[0]) == 1
                losses['loss_cls'] = weighted_cross_entropy(
                    cls_score, labels, label_weights, reduce=reduce)
                losses['acc'] = accuracy(cls_score, labels)
            else:
                # cls_score = cls_score.sigmoid()
                losses['loss_person_cls'] = weighted_binary_cross_entropy(
                    cls_score[:, 0], labels[:, 0] >= 1, label_weights)
                pos_inds = torch.nonzero(labels[:, 0] > 0).squeeze(1)
                losses['loss_action_cls'] = weighted_multilabel_binary_cross_entropy(
                    cls_score[pos_inds, 1:], labels[pos_inds, :], class_weights[pos_inds, 1:])
                acc, recall_thr, prec_thr, recall_k, prec_k = multilabel_accuracy(
                    cls_score, labels, topk=(3,5), thr=0.5)
                losses['acc'] = acc
                losses['recall@thr=0.5'] = recall_thr
                losses['prec@thr=0.5'] = prec_thr
                losses['recall@top3'] = recall_k[0]
                losses['prec@top3'] = prec_k[0]
                losses['recall@top5'] = recall_k[1]
                losses['prec@top5'] = prec_k[1]
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_inds = labels[:, 0] > 0
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_inds = labels > 0
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_reg'] = weighted_smoothl1(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None,
                       crop_quadruple=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if not self.multilabel_classification:
            scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            scores = cls_score.sigmoid() if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
            # TODO: add clip here

        def _bbox_crop_undo(bboxes, crop_quadruple):
            assert bboxes.shape[-1] % 4 == 0
            assert crop_quadruple is not None
            decropped = bboxes.clone()
            x1, y1, tw, th = crop_quadruple
            decropped[..., 0::2] = bboxes[..., 0::2] + x1
            decropped[..., 1::2] = bboxes[..., 1::2] + y1
            return decropped

        if crop_quadruple is not None:
            bboxes = _bbox_crop_undo(bboxes, crop_quadruple)

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            # TODO: apply class-agnostic nms
            if self.nms_class_agnostic:
                det_bboxes, det_labels = singleclass_nms(
                    bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            else:
                det_bboxes, det_labels = multiclass_nms(
                    bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.
        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.
        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.
        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
