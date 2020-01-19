import logging
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from ...registry import BACKBONES
from ..utils.resnet_r3d_utils import *


class BasicBlock(nn.Module):
    def __init__(self,
                 input_filters,
                 num_filters,
                 base_filters,
                 down_sampling=False,
                 down_sampling_temporal=None,
                 block_type='3d',
                 is_real_3d=True,
                 group=1,
                 with_bn=True):


        super(BasicBlock, self).__init__()
        self.num_filters = num_filters
        self.base_filters = base_filters
        self.input_filters = input_filters
        self.with_bn = with_bn
        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

        if block_type == '2.5d':
            assert is_real_3d
        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                self.down_sampling_stride = [2, 2, 2]
            else:
                self.down_sampling_stride = [1, 2, 2]
        else:
            self.down_sampling_stride = [1, 1, 1]

        self.down_sampling = down_sampling

        self.relu = nn.ReLU()
        self.conv1 = add_conv3d(input_filters, num_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=self.down_sampling_stride,
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn1 = add_bn(num_filters)
        self.conv2 = add_conv3d(num_filters, num_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=[1, 1, 1],
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn2 = add_bn(num_filters)
        if num_filters != input_filters or down_sampling:
            self.conv3 = conv3d(input_filters, num_filters, kernel=[1, 1, 1],
                                stride=self.down_sampling_stride, pad=[0, 0, 0])
            if self.with_bn:
                self.bn3 = nn.BatchNorm3d(num_filters, eps=1e-3)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.with_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_bn:
            out = self.bn2(out)

        if self.down_sampling or self.num_filters != self.input_filters:
            identity = self.conv3(identity)
            if self.with_bn:
                identity = self.bn3(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self,
                 input_filters,
                 num_filters,
                 base_filters,
                 down_sampling=False,
                 down_sampling_temporal=None,
                 block_type='3d',
                 is_real_3d=True,
                 group=1,
                 with_bn=True):

        super(Bottleneck, self).__init__()
        self.num_filters = num_filters
        self.base_filters = base_filters
        self.input_filters = input_filters
        self.with_bn = with_bn
        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

        if block_type == '2.5d':
            assert is_real_3d
        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                self.down_sampling_stride = [2, 2, 2]
            else:
                self.down_sampling_stride = [1, 2, 2]
        else:
            self.down_sampling_stride = [1, 1, 1]

        self.down_sampling = down_sampling
        self.relu = nn.ReLU()

        self.conv0 = add_conv3d(input_filters, base_filters, kernel=[
                                1, 1, 1], stride=[1, 1, 1], pad=[0, 0, 0], with_bn=self.with_bn)
        if self.with_bn:
            self.bn0 = add_bn(base_filters)

        self.conv1 = add_conv3d(base_filters, base_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=self.down_sampling_stride,
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn1 = add_bn(base_filters)

        self.conv2 = add_conv3d(base_filters, num_filters, kernel=[
                                1, 1, 1], pad=[0, 0, 0], stride=[1, 1, 1], with_bn=self.with_bn)

        if self.with_bn:
            self.bn2 = add_bn(num_filters)

        if num_filters != input_filters or down_sampling:
            self.conv3 = conv3d(input_filters, num_filters, kernel=[1, 1, 1],
                                stride=self.down_sampling_stride, pad=[0, 0, 0])
            if self.with_bn:
                self.bn3 = nn.BatchNorm3d(num_filters, eps=1e-3)

    def forward(self, x):
        identity = x
        if self.with_bn:
            out = self.relu(self.bn0(self.conv0(x)))
            out = self.relu(self.bn1(self.conv1(out)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.relu(self.conv0(x))
            out = self.relu(self.conv1(out))
            out = self.conv2(out)

        if self.down_sampling or self.num_filters != self.input_filters:
            identity = self.conv3(identity)
            if self.with_bn:
                identity = self.bn3(identity)

        out += identity
        out = self.relu(out)
        return out


def make_plain_res_layer(block, num_blocks, in_filters, num_filters, base_filters,
                         block_type='3d', down_sampling=False, down_sampling_temporal=None,
                         is_real_3d=True, with_bn=True):
    layers = []
    layers.append(block(in_filters, num_filters, base_filters, down_sampling=down_sampling,
                        down_sampling_temporal=down_sampling_temporal, block_type=block_type,
                        is_real_3d=is_real_3d, with_bn=with_bn))
    for i in range(num_blocks - 1):
        layers.append(block(num_filters, num_filters, base_filters,
                            block_type=block_type, is_real_3d=is_real_3d, with_bn=with_bn))
    return module_list(layers)


BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}
SHALLOW_FILTER_CONFIG = [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512]
]
DEEP_FILTER_CONFIG = [
    [256, 64],
    [512, 128],
    [1024, 256],
    [2048, 512]
]


@BACKBONES.register_module
class ResNet_R3D(nn.Module):
    def __init__(self,
                 pretrained=None,
                 num_input_channels=3,
                 depth=34,
                 block_type='2.5d',
                 channel_multiplier=1.0,
                 bottleneck_multiplier=1.0,
                 conv1_kernel_t=3,
                 conv1_stride_t=1,
                 use_pool1=False,
                 bn_eval=True,
                 bn_frozen=True,
                 with_bn=True):
        #         parameter initialization
        super(ResNet_R3D, self).__init__()
        self.pretrained = pretrained
        self.num_input_channels = num_input_channels
        self.depth = depth
        self.block_type = block_type
        self.channel_multiplier = channel_multiplier
        self.bottleneck_multiplier = bottleneck_multiplier
        self.conv1_kernel_t = conv1_kernel_t
        self.conv1_stride_t = conv1_stride_t
        self.use_pool1 = use_pool1
        self.relu = nn.ReLU()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_bn = with_bn
        global comp_count, comp_idx
        comp_idx = 0
        comp_count = 0

        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

#         stem block
        if self.block_type in ['2.5d', '2.5d-sep']:
            self.conv1_s = conv3d(self.num_input_channels, 45, [
                                  1, 7, 7], [1, 2, 2], [0, 3, 3])
            if self.with_bn:
                self.bn1_s = nn.BatchNorm3d(45, eps=1e-3)
            self.conv1_t = conv3d(45, 64, [self.conv1_kernel_t, 1, 1], [self.conv1_stride_t, 1, 1],
                                  [(self.conv1_kernel_t - 1) // 2, 0, 0])
            if self.with_bn:
                self.bn1_t = nn.BatchNorm3d(64, eps=1e-3)
        else:
            self.conv1 = conv3d(self.num_input_channels, 64, [self.conv1_kernel_t, 7, 7],
                                [self.conv1_stride_t, 2, 2], [(self.conv1_kernel_t - 1) // 2, 3, 3])
            if self.with_bn:
                self.bn1 = nn.BatchNorm3d(64, eps=1e-3)

        if self.use_pool1:
            self.pool1 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[
                                      1, 2, 2], padding=[0, 1, 1])

        self.stage_blocks = BLOCK_CONFIG[self.depth]
        if self.depth <= 18 or self.depth == 34:
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        if self.depth <= 34:
            self.filter_config = SHALLOW_FILTER_CONFIG
        else:
            self.filter_config = DEEP_FILTER_CONFIG
        self.filter_config = np.multiply(
            self.filter_config, self.channel_multiplier).astype(np.int)

        layer1 = make_plain_res_layer(self.block, self.stage_blocks[0],
                                      64, self.filter_config[0][0],
                                      int(self.filter_config[0][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type,
                                      with_bn=self.with_bn)
        self.add_module('layer1', layer1)
        layer2 = make_plain_res_layer(self.block, self.stage_blocks[1],
                                      self.filter_config[0][0], self.filter_config[1][0],
                                      int(self.filter_config[1][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer2', layer2)
        layer3 = make_plain_res_layer(self.block, self.stage_blocks[2],
                                      self.filter_config[1][0], self.filter_config[2][0],
                                      int(self.filter_config[2][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer3', layer3)
        layer4 = make_plain_res_layer(self.block, self.stage_blocks[3],
                                      self.filter_config[2][0], self.filter_config[3][0],
                                      int(self.filter_config[3][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer4', layer4)
        self.res_layers = ['layer1', 'layer2', 'layer3', 'layer4']

    def forward(self, x):
        if self.block_type in ['2.5d', '2.5d-sep']:
            if self.with_bn:
                x = self.relu(self.bn1_s(self.conv1_s(x)))
                x = self.relu(self.bn1_t(self.conv1_t(x)))
            else:
                x = self.relu(self.conv1_s(x))
                x = self.relu(self.conv1_t(x))
        else:
            if self.with_bn:
                x = self.relu(self.bn1(self.conv1(x)))
            else:
                x = self.relu(self.conv1(x))

        if self.use_pool1:
            x = self.pool1(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        return x

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(ResNet_R3D, self).train(mode)
        if self.bn_eval and self.with_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
