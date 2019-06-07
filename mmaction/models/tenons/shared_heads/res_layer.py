import logging

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..backbones import ResNet
from ..backbones.resnet import make_res_layer
from ...registry import HEADS
from ..spatial_temporal_modules.non_local import NonLocalModule


@HEADS.register_module
class ResLayer(nn.Module):

    def __init__(self,
                 depth,
                 pretrained=None,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 bn_eval=True,
                 bn_frozen=True,
                 all_frozen=False,
                 with_cp=False):
        super(ResLayer, self).__init__()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.all_frozen = all_frozen
        self.stage = stage
        block, stage_blocks = ResNet.arch_settings[depth]
        self.pretrained = pretrained
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.bn_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            for m in res_layer:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if self.all_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            res_layer.eval()
            for param in mod.parameters():
                param.requires_grad = False
