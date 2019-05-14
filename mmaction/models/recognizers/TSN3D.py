from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS


@RECOGNIZERS.register_module
class TSN3D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 flownet=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(TSN3D, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if flownet is not None:
            self.flownet = builder.build_flownet(flownet)

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(spatial_temporal_module)
        else:
            raise NotImplementedError

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(segmental_consensus)
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

    
    def extract_feat(self, img_group,
                     trajectory_forward=None,
                     trajectory_backward=None):
        x = self.backbone(img_group,
                          trajectory_forward=trajectory_forward,
                          trajectory_backward=trajectory_backward)
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
                img_forward = img_forward.transpose(1, 2).contiguous().view(
                    (img_forward.size(0), -1, img_forward.size(3), img_forward.size(4)))
                trajectory_forward = self.flownet(img_forward)
                img_backward = img_group.flip(2)[:, :, 1:, :, :]
                img_backward = img_backward.transpose(1, 2).contiguous().view(
                    (img_backward.size(0), -1, img_backward.size(3), img_backward.size(4)))
                trajectory_backward = self.flownet(img_backward)
            else:
                raise NotImplementedError
            x = self.extract_feat(img_group[:, :, 1:-1, :, :],
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
            losses.update(self.flownet.loss())
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
                img_forward = img_forward.transpose(1, 2).contiguous().view(
                    (img_forward.size(0), -1, img_forward.size(3), img_forward.size(4)))
                trajectory_forward = self.flownet(img_forward)
                img_backward = img_group.flip(2)[:, :, 1:, :, :]
                img_backward = img_backward.transpose(1, 2).contiguous().view(
                    (img_backward.size(0), -1, img_backward.size(3), img_backward.size(4)))
                trajectory_backward = self.flownet(img_backward)
            else:
                raise NotImplementedError
            x = self.extract_feat(img_group[:, :, 1:-1, :, :],
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
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy() 
        

