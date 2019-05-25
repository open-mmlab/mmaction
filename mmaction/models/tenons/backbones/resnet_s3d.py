import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ....utils.misc import rgetattr, rhasattr
from .resnet import ResNet 
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ....ops.trajectory_conv_package.traj_conv import TrajConv
from .. import flownets


from ...registry import BACKBONES

def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)


def conv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1, bias=False):
    "3x1x1 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3,1,1),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(dilation,0,0),
        dilation=dilation,
        bias=bias)

def trajconv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1, bias=False):
    "3x1x1 convolution with padding"
    return TrajConv(
        in_planes,
        out_planes,
        kernel_size=(3,1,1),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(dilation,0,0),
        dilation=dilation,
        bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 with_cp=False,
                 with_trajectory=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, spatial_stride, 1, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.if_inflate = if_inflate

        if self.if_inflate:
            self.conv1_t = conv3x1x1(planes, planes, 1, temporal_stride, dilation, bias=True)
            self.bn1_t = nn.BatchNorm3d(planes)
            if with_trajectory:
                self.conv2_t = trajconv3x1x1(planes, planes, bias=True)
            else:
                self.conv2_t = conv3x1x1(planes, planes, bias=True)
            self.bn2_t = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        assert not with_cp

        self.with_trajectory = with_trajectory

    def forward(self, input):
        x, traj_src = input

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.if_inflate:
            out = self.conv1_t(out)
            out = self.bn1_t(out)
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.if_inflate:
            out = self.relu(out)
            if self.with_trajectory:
                assert traj_src[0] is not None
                out = self.conv2_t(out, traj_src[0])
            else:
                out = self.conv2_t(out)
            out = self.bn2_t(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, traj_src[1:]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 with_cp=False,
                 with_trajectory=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=1,
            stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
            bias=False)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1,3,3),
            stride=(1, self.conv2_stride, self.conv2_stride),
            padding=(0, dilation, dilation),
            dilation=(1, dilation, dilation),
            bias=False)

        self.if_inflate = if_inflate
        if self.if_inflate:
            self.conv2_t = nn.Conv3d(
                planes,
                planes,
                kernel_size=(3,1,1),
                stride=(self.conv2_stride_t,1,1),
                padding=(1,0,0),
                dilation=1,
                bias=True)
            self.bn2_t = nn.BatchNorm3d(planes)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp

        self.with_trajectory = with_trajectory

    def forward(self, x):

        def _inner_forward(xx):
            x, traj_src = xx
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            if self.if_inflate:
                if self.with_trajectory:
                    assert traj_src is not None
                    out = self.conv2_t(out, traj_src[0])
                else:
                    out = self.conv2_t(out)
                out = self.bn2_t(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out, traj_src[1:]

        if self.with_cp and x.requires_grad:
            out, traj_remains = cp.checkpoint(_inner_forward, x)
        else:
            out, traj_remains = _inner_forward(x)

        out = self.relu(out)

        return out, traj_remains


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   with_cp=False,
                   traj_src_indices=-1):
    traj_src_indices = traj_src_indices if not isinstance(traj_src_indices, int) else (traj_src_indices, ) * blocks
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * blocks
    assert len(inflate_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            nn.BatchNorm3d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            if_inflate=(inflate_freq[0] == 1),
            with_trajectory=(traj_src_indices[0]>-1),
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
                with_trajectory=(traj_src_indices[i]>-1),
                with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet_S3D(nn.Module):
    """ResNet_S3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
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
                 pretrained=None,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel_t=5,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 use_pool2=True,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_freq=(1, 1, 1, 1),    # For C2D baseline, this is set to -1.
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False,
                 with_trajectory=False,
                 trajectory_source_indices=-1,
                 trajectory_downsample_method='ave',
                 conv_bias=0.2):
        super(ResNet_S3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * num_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.with_trajectory = with_trajectory
        self.trajectory_source_indices = trajectory_source_indices \
            if not isinstance(trajectory_source_indices, int) else [trajectory_source_indices, ] * num_stages
        self.trajectory_downsample_method = trajectory_downsample_method

        self.conv_bias = conv_bias

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        for stage in range(num_stages):
            self.trajectory_source_indices[stage] = self.trajectory_source_indices[stage] \
                if not isinstance(self.trajectory_source_indices[stage], int) else (self.trajectory_source_indices[stage], ) * self.stage_blocks[stage]
        self.inplanes = 64

        if conv1_kernel_t > 1:
            self.conv1 = nn.Conv3d(
                3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
            self.conv1_t = nn.Conv3d(
                64, 64, kernel_size=(conv1_kernel_t,1,1), stride=(conv1_stride_t,1,1), padding=((conv1_kernel_t-1)//2,1,1), bias=True)
            self.bn1_t = nn.BatchNorm3d(64)
        else:
            self.conv1 = nn.Conv3d(
                3, 64, kernel_size=(1,7,7), stride=(conv1_stride_t,2,2), padding=(0,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t,3,3), stride=(pool1_stride_t,2,2), padding=(pool1_kernel_t//2,1,1))
        self.use_pool2 = use_pool2
        if self.use_pool2:
            self.pool2 = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            traj_src_indices = self.trajectory_source_indices[i] \
                if not isinstance(self.trajectory_source_indices[i], int) \
                else (self.trajectory_source_indices[i], ) * num_blocks
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate_freq=self.inflate_freqs[i],
                with_cp=with_cp,
                traj_src_indices=traj_src_indices)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            resnet2d = ResNet(self.depth)
            load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d) or isinstance(module, TrajConv):
                    if rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                    else:
                        kaiming_init(module, bias=self.conv_bias)
                elif isinstance(module, nn.BatchNorm3d):
                    if rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
                    else:
                        constant_init(module, 1)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m, bias=self.conv_bias)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, trajectory_forward=None, trajectory_backward=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            y = []
            for j in self.trajectory_source_indices[i]:
                if j > -1:
                    flow_forward = trajectory_forward[j]  ## N, 2*T, H, W (..x3y3x4y4..)
                    flow_backward = trajectory_backward[j]
                    flow_forward = flow_forward.view((flow_forward.size(0), -1, 2, flow_forward.size(2), flow_forward.size(3)))
                    flow_backward = flow_backward.view((flow_backward.size(0), -1, 2, flow_backward.size(2), flow_backward.size(3)))
                    flow_forward_x, flow_forward_y = torch.split(flow_forward, 1, 2)
                    flow_backward_x, flow_backward_y = torch.split(flow_backward, 1, 2)
                    flow_backward_x = flow_backward_x.flip(1).view((flow_backward_x.size(0), 1, flow_backward_x.size(1),
                                                                    flow_backward_x.size(3), flow_backward_x.size(4)))  # N,T,1,H,W => N,1,T,H,W
                    flow_backward_y = flow_backward_y.flip(1).view((flow_backward_y.size(0), 1, flow_backward_y.size(1),
                                                                    flow_backward_y.size(3), flow_backward_y.size(4)))
                    flow_forward_x = flow_forward_x.view((flow_forward_x.size(0), 1, flow_forward_x.size(1),
                                                          flow_forward_x.size(3), flow_forward_x.size(4)))
                    flow_forward_y = flow_forward_y.view((flow_forward_y.size(0), 1, flow_forward_y.size(1),
                                                          flow_forward_y.size(3), flow_forward_y.size(4)))
                    flow_zero = torch.zeros_like(flow_forward_x)
                    y.append(torch.cat((flow_backward_y, flow_backward_x, flow_zero, flow_zero, flow_forward_y, flow_forward_x), 1))
                else:
                    y.append(None)
            
            x, remains = res_layer((x, y))
            assert len(remains) == 0 ## TODO: delete if check passes
            if i in self.out_indices:
                outs.append(x)
            if self.use_pool2 and i == 0:
                x = self.pool2(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_S3D, self).train(mode)
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
