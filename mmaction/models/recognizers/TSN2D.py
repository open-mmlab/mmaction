import torch.nn as nn
from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS


@RECOGNIZERS.register_module
class TSN2D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 modality='RGB',
                 in_channels=3,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(TSN2D, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.modality = modality
        self.in_channels = in_channels

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

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

        if modality == 'Flow' or modality == 'RGBDiff':
            self._construct_2d_backbone_conv1(in_channels)

    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def _construct_2d_backbone_conv1(self, in_channels):
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
        new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
            new_kernel_size).contiguous()  # make contiguous!

        new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,
                                   conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                                   bias=True if len(params) == 2 else False)
        new_conv_layer.weight.data = new_kernel_data
        if len(params) == 2:
            new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv_layer)

    def init_weights(self):
        super(TSN2D, self).init_weights()
        self.backbone.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

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
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + x.shape[1:])
        if self.with_segmental_consensus:
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        losses = dict()
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
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + x.shape[1:])
        if self.with_segmental_consensus:
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy()
