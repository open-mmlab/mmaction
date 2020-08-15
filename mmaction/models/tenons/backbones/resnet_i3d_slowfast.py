import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ....utils.misc import rgetattr, rhasattr
from .resnet import ResNet
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..utils.nonlocal_block import build_nonlocal_block
from ..spatial_temporal_modules.non_local import NonLocalModule

from ...registry import BACKBONES
from .resnet_i3d import conv3x3x3, conv1x3x3, BasicBlock, Bottleneck


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   # from another pathway
                   lateral_inplanes=0,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   with_cp=False):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq, ) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(
                inplanes + lateral_inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            nn.BatchNorm3d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes + lateral_inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            if_inflate= (inflate_freq[0] == 1),
            inflate_style=inflate_style,
            nonlocal_cfg=nonlocal_cfg if nonlocal_freq[0] == 1 else None,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                planes,
                1, 1,
                dilation,
                style=style,
                if_inflate= (inflate_freq[i] == 1),
                inflate_style=inflate_style,
                nonlocal_cfg=nonlocal_cfg if nonlocal_freq[i] == 1 else None,
                with_cp=with_cp))

    return nn.Sequential(*layers)


class pathway(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 # reduce base channel by
                 channel_mul_inv=1,
                 lateral=True,
                 # alpha
                 alpha=8,
                 # beta inv
                 beta_inv=8,
                 lateral_type='conv',
                 lateral_op='concat',
                 conv1_kernel_t=1,
                 fusion_kernel_size=5,
                 spatial_strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 inflate_freqs=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1, ),
                 nonlocal_freqs=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 with_cp=False):
        super(pathway, self).__init__()
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64 // channel_mul_inv
        if lateral:
            if lateral_type == 'toC':
                lateral_inplanes = self.inplanes * alpha // beta_inv
            elif lateral_type == 'sampling':
                lateral_inplanes = self.inplanes // beta_inv
            elif lateral_type == 'conv':
                lateral_inplanes = self.inplanes * 2 // beta_inv
                self.conv1_lateral = nn.Conv3d(self.inplanes // beta_inv,
                    self.inplanes * 2 // beta_inv,
                    kernel_size=(fusion_kernel_size, 1, 1),
                    stride=(alpha, 1, 1),
                    padding=((fusion_kernel_size - 1) // 2, 0, 0),
                    bias=False)
            else:
                raise NotImplementedError
        else:
            lateral_inplanes = 0

        self.conv1 = nn.Conv3d(3, 64 // channel_mul_inv,
                            kernel_size=(conv1_kernel_t, 7, 7),
                            stride=(1, 2, 2),
                            padding=((conv1_kernel_t-1)//2, 3, 3),
                            bias=False)

        self.bn1 = nn.BatchNorm3d(64 // channel_mul_inv)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.res_layers = []
        self.lateral_connections = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = 1    # all temporal strides are set to 1 in SlowFast
            dilation = dilations[i]
            planes = 64 * 2**i // channel_mul_inv
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                lateral_inplanes=lateral_inplanes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=style,
                inflate_freq=inflate_freqs[i],
                inflate_style=inflate_style,
                nonlocal_freq=nonlocal_freqs[i],
                nonlocal_cfg=nonlocal_cfg if i in nonlocal_stages else None,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            if lateral:
                if lateral_type == 'toC':
                    lateral_inplanes = self.inplanes * alpha // beta_inv
                elif lateral_type == 'sampling':
                    lateral_inplanes = self.inplanes // beta_inv
                elif lateral_type == 'conv':
                    lateral_inplanes = self.inplanes * 2 // beta_inv
                    lateral_name = 'layer{}_lateral'.format(i + 1)
                    setattr(self, lateral_name,
                            nn.Conv3d(self.inplanes // beta_inv,
                            self.inplanes * 2 // beta_inv,
                            kernel_size=(fusion_kernel_size, 1, 1),
                            stride=(alpha, 1, 1),
                            padding=((fusion_kernel_size - 1) // 2, 0, 0),
                            bias=False))
                    self.lateral_connections.append(lateral_name)
            else:
                lateral_inplanes = 0

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)


@BACKBONES.register_module
class ResNet_I3D_SlowFast(nn.Module):
    """ResNet_I3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        alpha (int): The frame ratio between fast and slow pathways.
            The original parameter tau in pySlowFast is removed. In the original
            paper, it says: "Our Fast pathway works with a small temporal stride
            of τ/α, where α > 1 is the frame rate ratio between the Fast and
            Slow pathways." Here we force τ/α to be 1 and adjust `new_length`
            and `new_step` in dataset accordingly.
        beta_inv (int): The channel width ratio between slow and fast pathways.
        pretrained_slow (str): Path of 2D pretrained weights for slow pathway.
        pretrained_fast (str): Path of 2D pretrained weights for fast pathway.
        num_stages (int): Resnet stages, normally 4.
        spatial_strides (Sequence[int]): Spatial strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 alpha=8,
                 beta_inv=8,
                 pretrained_slow=None,
                 pretrained_fast=None,
                 num_stages=4,
                 lateral_type='conv',
                 lateral_op='concat',
                 spatial_strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 slow_conv1_kernel_t=1,
                 fast_conv1_kernel_t=5,
                 style='pytorch',
                 frozen_stages=-1,
                 slow_inflate_freq=(0, 0, 1, 1),
                 fast_inflate_freq=(1, 1, 1, 1),
                 fusion_kernel_size=5,
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1, ),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False):
        super(ResNet_I3D_SlowFast, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.alpha = alpha
        self.beta_inv = beta_inv
        # self.pretrained = pretrained
        self.pretrained_slow = pretrained_slow
        self.pretrained_fast = pretrained_fast
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.lateral_type = lateral_type
        self.lateral_op = lateral_op
        assert lateral_type in ['conv']   # in ['toC', 'sampling', 'conv']
        assert lateral_op in ['concat']   # in ['sum', 'concat']
        self.spatial_strides = spatial_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.slow_inflate_freq = slow_inflate_freq
        if isinstance(slow_inflate_freq, int):
            self.slow_inflate_freq = (slow_inflate_freq, ) * num_stages
        self.fast_inflate_freq = fast_inflate_freq
        if isinstance(fast_inflate_freq, int):
            self.fast_inflate_freq = (fast_inflate_freq, ) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq, ) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.fusion_kernel_size = fusion_kernel_size

        self.slow_path = pathway(depth, num_stages=num_stages, channel_mul_inv=1,
                                 lateral=True,
                                 alpha=alpha,
                                 beta_inv=beta_inv,
                                 lateral_type=lateral_type,
                                 lateral_op=lateral_op,
                                 conv1_kernel_t=slow_conv1_kernel_t,
                                 spatial_strides=spatial_strides,
                                 dilations=dilations,
                                 style=style,
                                 inflate_freqs=self.slow_inflate_freq,
                                 inflate_style=inflate_style,
                                 nonlocal_stages=nonlocal_stages,
                                 nonlocal_freqs=nonlocal_freq,
                                 nonlocal_cfg=nonlocal_cfg,
                                 with_cp=with_cp)
        self.fast_path = pathway(depth, num_stages=num_stages, channel_mul_inv=beta_inv,
                                 lateral=False,
                                 conv1_kernel_t=fast_conv1_kernel_t,
                                 spatial_strides=spatial_strides,
                                 dilations=dilations,
                                 style=style,
                                 inflate_freqs=self.fast_inflate_freq,
                                 inflate_style=inflate_style,
                                 nonlocal_stages=nonlocal_stages,
                                 nonlocal_freqs=nonlocal_freq,
                                 nonlocal_cfg=nonlocal_cfg,
                                 with_cp=with_cp)

    def init_weights(self):
        logger = logging.getLogger()
        if self.pretrained_slow:
            resnet2d = ResNet(self.depth)
            load_checkpoint(resnet2d, self.pretrained_slow, strict=False, logger=logger)
            for name, module in self.slow_path.named_modules():
                if isinstance(module, NonLocalModule):
                    module.init_weights()
                elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                    old_weight = rgetattr(resnet2d, name).weight.data
                    old_shape = old_weight.shape
                    new_shape = module.weight.data.shape
                    if new_shape[1] != old_shape[1]:
                        new_ch = new_shape[1] - old_shape[1]
                        pad_shape = old_shape
                        pad_shape = pad_shape[:1] + (new_ch, ) + pad_shape[2:]
                        old_weight = torch.cat((old_weight, torch.zeros(pad_shape).type_as(old_weight).to(old_weight.device)), dim=1)


                    new_weight = old_weight.unsqueeze(2).expand_as(module.weight.data) / new_shape[2]
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
                    print(name)
        else:
            for m in self.slow_path.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

        if self.pretrained_fast:
            resnet2d = ResNet(self.depth, base_channels=64 // self.beta_inv)
            load_checkpoint(resnet2d, self.pretrained_fast, strict=False, logger=logger)
            for name, module in self.fast_path.named_modules():
                if isinstance(module, NonLocalModule):
                    module.init_weights()
                elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                    old_weight = rgetattr(resnet2d, name).weight.data
                    old_shape = old_weight.shape
                    new_shape = module.weight.data.shape
                    if new_shape[1] != old_shape[1]:
                        new_ch = new_shape[1] - old_shape[1]
                        pad_shape = old_shape
                        pad_shape = pad_shape[:1] + (new_ch, ) + pad_shape[2:]
                        old_weight = torch.cat((old_weight, torch.zeros(pad_shape).type_as(old_weight).to(old_weight.device)), dim=1)


                    new_weight = old_weight.unsqueeze(2).expand_as(module.weight.data) / new_shape[2]
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
                    print(name)
        else:
            for m in self.fast_path.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

    def forward(self, x):
        x_slow = self.slow_path.conv1(x[:, :, ::self.alpha])
        x_slow = self.slow_path.bn1(x_slow)
        x_slow = self.slow_path.relu(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = self.fast_path.conv1(x)
        x_fast = self.fast_path.bn1(x_fast)
        x_fast = self.fast_path.relu(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
        x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        outs = []
        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)

            if self.lateral_type == 'conv' and i != 3:
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

            if i in self.out_indices:
                outs.append((x_slow, x_fast))

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_I3D_SlowFast, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
