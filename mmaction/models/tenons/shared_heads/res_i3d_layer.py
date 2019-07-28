import logging

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..backbones import ResNet_I3D
from ..backbones.resnet_i3d import make_res_layer
from ...registry import HEADS


@HEADS.register_module
class ResI3DLayer(nn.Module):

    def __init__(self,
                 depth,
                 pretrained=None,
                 pretrained2d=True,
                 stage=3,
                 spatial_stride=2,
                 temporal_stride=1,
                 dilation=1,
                 style='pytorch',
                 inflate_freq=1,
                 inflate_style='3x1x1',
                 bn_eval=True,
                 bn_frozen=True,
                 all_frozen=False,
                 with_cp=False):
        super(ResI3DLayer, self).__init__()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.all_frozen = all_frozen
        self.stage = stage
        block, stage_blocks = ResNet_I3D.arch_settings[depth]
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        self.inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * stage

        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            dilation=dilation,
            style=style,
            inflate_freq=self.inflate_freq,
            inflate_style='3x1x1',
            with_cp=with_cp)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            if self.pretrained2d:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
                for name, module in self.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info("{}.weight loaded from weights file into {}".format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info("{}.bias loaded from weights file into {}".format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info("{}.{} loaded from weights file into {}".format(name, attr, getattr(rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
            else:
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
        super(ResI3DLayer, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.bn_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            for m in res_layer:
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if self.all_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            res_layer.eval()
            for param in res_layer.parameters():
                param.requires_grad = False
