import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ...registry import BACKBONES
# from mmaction.ops.reflection_pad3d import reflection_pad3d


__all__ = ['InceptionV1_I3D']

@BACKBONES.register_module
class InceptionV1_I3D(nn.Module):

    ## TODO:
    ## Refactor it into a more modular way
    ## Reference: Table 1 from https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf

    def __init__(self,
                 pretrained=None,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 modality='RGB'):
        super(InceptionV1_I3D, self).__init__()

        self.pretrained = pretrained
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.modality = modality

        inplace = True
        assert modality in ['RGB', 'Flow']
        if modality == 'RGB':
            self.conv1_7x7_s2 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        else:
            self.conv1_7x7_s2 = nn.Conv3d(2, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        self.conv1_7x7_s2_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv2_3x3_reduce_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv3d(64, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.conv2_3x3_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        ##########
        self.inception_3a_1x1 = nn.Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_1x1_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)

        self.inception_3a_branch1_3x3_reduce = nn.Conv3d(192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_branch1_3x3_reduce_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_3a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_branch1_3x3 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3a_branch1_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3a_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_3a_branch2_3x3_reduce = nn.Conv3d(192, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_branch2_3x3_reduce_bn = nn.BatchNorm3d(16, eps=1e-05, affine=True)
        self.inception_3a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_branch2_3x3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3a_branch2_3x3_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3a_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_3a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_3a_pool_proj = nn.Conv3d(192, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_pool_proj_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)

        self.inception_3b_1x1 = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_1x1_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)

        self.inception_3b_branch1_3x3_reduce = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_branch1_3x3_reduce_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_branch1_3x3 = nn.Conv3d(128, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3b_branch1_3x3_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_3b_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_3b_branch2_3x3_reduce = nn.Conv3d(256, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_branch2_3x3 = nn.Conv3d(32, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3b_branch2_3x3_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_3b_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_3b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_3b_pool_proj = nn.Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)

        self.inception_3c_pool = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        ##########
        self.inception_4a_1x1 = nn.Conv3d(480, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_1x1_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)

        self.inception_4a_branch1_3x3_reduce = nn.Conv3d(480, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_branch1_3x3_reduce_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_4a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_branch1_3x3 = nn.Conv3d(96, 208, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4a_branch1_3x3_bn = nn.BatchNorm3d(208, eps=1e-05, affine=True)
        self.inception_4a_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_4a_branch2_3x3_reduce = nn.Conv3d(480, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_branch2_3x3_reduce_bn = nn.BatchNorm3d(16, eps=1e-05, affine=True)
        self.inception_4a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_branch2_3x3 = nn.Conv3d(16, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4a_branch2_3x3_bn = nn.BatchNorm3d(48, eps=1e-05, affine=True)
        self.inception_4a_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_4a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4a_pool_proj = nn.Conv3d(480, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)

        self.inception_4b_1x1 = nn.Conv3d(512, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_1x1_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)

        self.inception_4b_branch1_3x3_reduce = nn.Conv3d(512, 112, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_branch1_3x3_reduce_bn = nn.BatchNorm3d(112, eps=1e-05, affine=True)
        self.inception_4b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_branch1_3x3 = nn.Conv3d(112, 224, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4b_branch1_3x3_bn = nn.BatchNorm3d(224, eps=1e-05, affine=True)
        self.inception_4b_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_4b_branch2_3x3_reduce = nn.Conv3d(512, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_branch2_3x3_reduce_bn = nn.BatchNorm3d(24, eps=1e-05, affine=True)
        self.inception_4b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_branch2_3x3 = nn.Conv3d(24, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4b_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4b_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_4b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4b_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)

        self.inception_4c_1x1 = nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_1x1_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)

        self.inception_4c_branch1_3x3_reduce = nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_branch1_3x3_reduce_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4c_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_branch1_3x3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4c_branch1_3x3_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_4c_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_4c_branch2_3x3_reduce = nn.Conv3d(512, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_branch2_3x3_reduce_bn = nn.BatchNorm3d(24, eps=1e-05, affine=True)
        self.inception_4c_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_branch2_3x3 = nn.Conv3d(24, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4c_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4c_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_4c_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4c_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)

        self.inception_4d_1x1 = nn.Conv3d(512, 112, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_1x1_bn = nn.BatchNorm3d(112, eps=1e-05, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)

        self.inception_4d_branch1_3x3_reduce = nn.Conv3d(512, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_branch1_3x3_reduce_bn = nn.BatchNorm3d(144, eps=1e-05, affine=True)
        self.inception_4d_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_branch1_3x3 = nn.Conv3d(144, 288, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4d_branch1_3x3_bn = nn.BatchNorm3d(288, eps=1e-05, affine=True)
        self.inception_4d_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_4d_branch2_3x3_reduce = nn.Conv3d(512, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_4d_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_branch2_3x3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4d_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4d_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_4d_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4d_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)

        self.inception_4e_1x1 = nn.Conv3d(528, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_1x1_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_4e_relu_1x1 = nn.ReLU(inplace)

        self.inception_4e_branch1_3x3_reduce = nn.Conv3d(528, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_branch1_3x3_reduce_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_4e_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_branch1_3x3 = nn.Conv3d(160, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4e_branch1_3x3_bn = nn.BatchNorm3d(320, eps=1e-05, affine=True)
        self.inception_4e_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_4e_branch2_3x3_reduce = nn.Conv3d(528, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_4e_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_branch2_3x3 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4e_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4e_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_4e_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4e_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4e_pool_proj = nn.Conv3d(528, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4e_relu_pool_proj = nn.ReLU(inplace)

        self.inception_4f_pool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        ##########
        self.inception_5a_1x1 = nn.Conv3d(832, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_1x1_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)

        self.inception_5a_branch1_3x3_reduce = nn.Conv3d(832, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_branch1_3x3_reduce_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_5a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_branch1_3x3 = nn.Conv3d(160, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5a_branch1_3x3_bn = nn.BatchNorm3d(320, eps=1e-05, affine=True)
        self.inception_5a_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_5a_branch2_3x3_reduce = nn.Conv3d(832, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_5a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_branch2_3x3 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5a_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5a_branch2_relu_3x3 = nn.ReLU(inplace)

        # self.inception_5a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_5a_pool_proj = nn.Conv3d(832, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)

        self.inception_5b_1x1 = nn.Conv3d(832, 384, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_1x1_bn = nn.BatchNorm3d(384, eps=1e-05, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)

        self.inception_5b_branch1_3x3_reduce = nn.Conv3d(832, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_branch1_3x3_reduce_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_5b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_branch1_3x3 = nn.Conv3d(192, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5b_branch1_3x3_bn = nn.BatchNorm3d(384, eps=1e-05, affine=True)
        self.inception_5b_branch1_relu_3x3 = nn.ReLU(inplace)

        self.inception_5b_branch2_3x3_reduce = nn.Conv3d(832, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_branch2_3x3_reduce_bn = nn.BatchNorm3d(48, eps=1e-05, affine=True)
        self.inception_5b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_branch2_3x3 = nn.Conv3d(48, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5b_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5b_branch2_relu_3x3 = nn.ReLU(inplace)

        self.inception_5b_pool = nn.MaxPool3d(3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv3d(832, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05,  affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)


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


    def forward(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(F.pad(input, (2, 4, 2, 4, 2, 4)))
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)

        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_branch1_3x3_reduce_out = self.inception_3a_branch1_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_branch1_3x3_reduce_bn_out = self.inception_3a_branch1_3x3_reduce_bn(inception_3a_branch1_3x3_reduce_out)
        inception_3a_branch1_relu_3x3_reduce_out = self.inception_3a_branch1_relu_3x3_reduce(inception_3a_branch1_3x3_reduce_bn_out)
        inception_3a_branch1_3x3_out = self.inception_3a_branch1_3x3(inception_3a_branch1_3x3_reduce_bn_out)
        inception_3a_branch1_3x3_bn_out = self.inception_3a_branch1_3x3_bn(inception_3a_branch1_3x3_out)
        inception_3a_branch1_relu_3x3_out = self.inception_3a_branch1_relu_3x3(inception_3a_branch1_3x3_bn_out)
        inception_3a_branch2_3x3_reduce_out = self.inception_3a_branch2_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_branch2_3x3_reduce_bn_out = self.inception_3a_branch2_3x3_reduce_bn(inception_3a_branch2_3x3_reduce_out)
        inception_3a_branch2_relu_3x3_reduce_out = self.inception_3a_branch2_relu_3x3_reduce(inception_3a_branch2_3x3_reduce_bn_out)
        inception_3a_branch2_3x3_out = self.inception_3a_branch2_3x3(inception_3a_branch2_3x3_reduce_bn_out)
        inception_3a_branch2_3x3_bn_out = self.inception_3a_branch2_3x3_bn(inception_3a_branch2_3x3_out)
        inception_3a_branch2_relu_3x3_out = self.inception_3a_branch2_relu_3x3(inception_3a_branch2_3x3_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out,inception_3a_branch1_3x3_bn_out,inception_3a_branch2_3x3_bn_out,inception_3a_pool_proj_bn_out], 1)


        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_branch1_3x3_reduce_out = self.inception_3b_branch1_3x3_reduce(inception_3a_output_out)
        inception_3b_branch1_3x3_reduce_bn_out = self.inception_3b_branch1_3x3_reduce_bn(inception_3b_branch1_3x3_reduce_out)
        inception_3b_branch1_relu_3x3_reduce_out = self.inception_3b_branch1_relu_3x3_reduce(inception_3b_branch1_3x3_reduce_bn_out)
        inception_3b_branch1_3x3_out = self.inception_3b_branch1_3x3(inception_3b_branch1_3x3_reduce_bn_out)
        inception_3b_branch1_3x3_bn_out = self.inception_3b_branch1_3x3_bn(inception_3b_branch1_3x3_out)
        inception_3b_branch1_relu_3x3_out = self.inception_3b_branch1_relu_3x3(inception_3b_branch1_3x3_bn_out)
        inception_3b_branch2_3x3_reduce_out = self.inception_3b_branch2_3x3_reduce(inception_3a_output_out)
        inception_3b_branch2_3x3_reduce_bn_out = self.inception_3b_branch2_3x3_reduce_bn(inception_3b_branch2_3x3_reduce_out)
        inception_3b_branch2_relu_3x3_reduce_out = self.inception_3b_branch2_relu_3x3_reduce(inception_3b_branch2_3x3_reduce_bn_out)
        inception_3b_branch2_3x3_out = self.inception_3b_branch2_3x3(inception_3b_branch2_3x3_reduce_bn_out)
        inception_3b_branch2_3x3_bn_out = self.inception_3b_branch2_3x3_bn(inception_3b_branch2_3x3_out)
        inception_3b_branch2_relu_3x3_out = self.inception_3b_branch2_relu_3x3(inception_3b_branch2_3x3_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out,inception_3b_branch1_3x3_bn_out,inception_3b_branch2_3x3_bn_out,inception_3b_pool_proj_bn_out], 1)


        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)

        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_pool_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_branch1_3x3_reduce_out = self.inception_4a_branch1_3x3_reduce(inception_3c_pool_out)
        inception_4a_branch1_3x3_reduce_bn_out = self.inception_4a_branch1_3x3_reduce_bn(inception_4a_branch1_3x3_reduce_out)
        inception_4a_branch1_relu_3x3_reduce_out = self.inception_4a_branch1_relu_3x3_reduce(inception_4a_branch1_3x3_reduce_bn_out)
        inception_4a_branch1_3x3_out = self.inception_4a_branch1_3x3(inception_4a_branch1_3x3_reduce_bn_out)
        inception_4a_branch1_3x3_bn_out = self.inception_4a_branch1_3x3_bn(inception_4a_branch1_3x3_out)
        inception_4a_branch1_relu_3x3_out = self.inception_4a_branch1_relu_3x3(inception_4a_branch1_3x3_bn_out)
        inception_4a_branch2_3x3_reduce_out = self.inception_4a_branch2_3x3_reduce(inception_3c_pool_out)
        inception_4a_branch2_3x3_reduce_bn_out = self.inception_4a_branch2_3x3_reduce_bn(inception_4a_branch2_3x3_reduce_out)
        inception_4a_branch2_relu_3x3_reduce_out = self.inception_4a_branch2_relu_3x3_reduce(inception_4a_branch2_3x3_reduce_bn_out)
        inception_4a_branch2_3x3_out = self.inception_4a_branch2_3x3(inception_4a_branch2_3x3_reduce_bn_out)
        inception_4a_branch2_3x3_bn_out = self.inception_4a_branch2_3x3_bn(inception_4a_branch2_3x3_out)
        inception_4a_branch2_relu_3x3_out = self.inception_4a_branch2_relu_3x3(inception_4a_branch2_3x3_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_pool_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out,inception_4a_branch1_3x3_bn_out,inception_4a_branch2_3x3_bn_out,inception_4a_pool_proj_bn_out], 1)

        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_branch1_3x3_reduce_out = self.inception_4b_branch1_3x3_reduce(inception_4a_output_out)
        inception_4b_branch1_3x3_reduce_bn_out = self.inception_4b_branch1_3x3_reduce_bn(inception_4b_branch1_3x3_reduce_out)
        inception_4b_branch1_relu_3x3_reduce_out = self.inception_4b_branch1_relu_3x3_reduce(inception_4b_branch1_3x3_reduce_bn_out)
        inception_4b_branch1_3x3_out = self.inception_4b_branch1_3x3(inception_4b_branch1_3x3_reduce_bn_out)
        inception_4b_branch1_3x3_bn_out = self.inception_4b_branch1_3x3_bn(inception_4b_branch1_3x3_out)
        inception_4b_branch1_relu_3x3_out = self.inception_4b_branch1_relu_3x3(inception_4b_branch1_3x3_bn_out)
        inception_4b_branch2_3x3_reduce_out = self.inception_4b_branch2_3x3_reduce(inception_4a_output_out)
        inception_4b_branch2_3x3_reduce_bn_out = self.inception_4b_branch2_3x3_reduce_bn(inception_4b_branch2_3x3_reduce_out)
        inception_4b_branch2_relu_3x3_reduce_out = self.inception_4b_branch2_relu_3x3_reduce(inception_4b_branch2_3x3_reduce_bn_out)
        inception_4b_branch2_3x3_out = self.inception_4b_branch2_3x3(inception_4b_branch2_3x3_reduce_bn_out)
        inception_4b_branch2_3x3_bn_out = self.inception_4b_branch2_3x3_bn(inception_4b_branch2_3x3_out)
        inception_4b_branch2_relu_3x3_out = self.inception_4b_branch2_relu_3x3(inception_4b_branch2_3x3_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out,inception_4b_branch1_3x3_bn_out,inception_4b_branch2_3x3_bn_out,inception_4b_pool_proj_bn_out], 1)

        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_branch1_3x3_reduce_out = self.inception_4c_branch1_3x3_reduce(inception_4b_output_out)
        inception_4c_branch1_3x3_reduce_bn_out = self.inception_4c_branch1_3x3_reduce_bn(inception_4c_branch1_3x3_reduce_out)
        inception_4c_branch1_relu_3x3_reduce_out = self.inception_4c_branch1_relu_3x3_reduce(inception_4c_branch1_3x3_reduce_bn_out)
        inception_4c_branch1_3x3_out = self.inception_4c_branch1_3x3(inception_4c_branch1_3x3_reduce_bn_out)
        inception_4c_branch1_3x3_bn_out = self.inception_4c_branch1_3x3_bn(inception_4c_branch1_3x3_out)
        inception_4c_branch1_relu_3x3_out = self.inception_4c_branch1_relu_3x3(inception_4c_branch1_3x3_bn_out)
        inception_4c_branch2_3x3_reduce_out = self.inception_4c_branch2_3x3_reduce(inception_4b_output_out)
        inception_4c_branch2_3x3_reduce_bn_out = self.inception_4c_branch2_3x3_reduce_bn(inception_4c_branch2_3x3_reduce_out)
        inception_4c_branch2_relu_3x3_reduce_out = self.inception_4c_branch2_relu_3x3_reduce(inception_4c_branch2_3x3_reduce_bn_out)
        inception_4c_branch2_3x3_out = self.inception_4c_branch2_3x3(inception_4c_branch2_3x3_reduce_bn_out)
        inception_4c_branch2_3x3_bn_out = self.inception_4c_branch2_3x3_bn(inception_4c_branch2_3x3_out)
        inception_4c_branch2_relu_3x3_out = self.inception_4c_branch2_relu_3x3(inception_4c_branch2_3x3_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out,inception_4c_branch1_3x3_bn_out,inception_4c_branch2_3x3_bn_out,inception_4c_pool_proj_bn_out], 1)

        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_branch1_3x3_reduce_out = self.inception_4d_branch1_3x3_reduce(inception_4c_output_out)
        inception_4d_branch1_3x3_reduce_bn_out = self.inception_4d_branch1_3x3_reduce_bn(inception_4d_branch1_3x3_reduce_out)
        inception_4d_branch1_relu_3x3_reduce_out = self.inception_4d_branch1_relu_3x3_reduce(inception_4d_branch1_3x3_reduce_bn_out)
        inception_4d_branch1_3x3_out = self.inception_4d_branch1_3x3(inception_4d_branch1_3x3_reduce_bn_out)
        inception_4d_branch1_3x3_bn_out = self.inception_4d_branch1_3x3_bn(inception_4d_branch1_3x3_out)
        inception_4d_branch1_relu_3x3_out = self.inception_4d_branch1_relu_3x3(inception_4d_branch1_3x3_bn_out)
        inception_4d_branch2_3x3_reduce_out = self.inception_4d_branch2_3x3_reduce(inception_4c_output_out)
        inception_4d_branch2_3x3_reduce_bn_out = self.inception_4d_branch2_3x3_reduce_bn(inception_4d_branch2_3x3_reduce_out)
        inception_4d_branch2_relu_3x3_reduce_out = self.inception_4d_branch2_relu_3x3_reduce(inception_4d_branch2_3x3_reduce_bn_out)
        inception_4d_branch2_3x3_out = self.inception_4d_branch2_3x3(inception_4d_branch2_3x3_reduce_bn_out)
        inception_4d_branch2_3x3_bn_out = self.inception_4d_branch2_3x3_bn(inception_4d_branch2_3x3_out)
        inception_4d_branch2_relu_3x3_out = self.inception_4d_branch2_relu_3x3(inception_4d_branch2_3x3_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out,inception_4d_branch1_3x3_bn_out,inception_4d_branch2_3x3_bn_out,inception_4d_pool_proj_bn_out], 1)

        inception_4e_1x1_out = self.inception_4e_1x1(inception_4d_output_out)
        inception_4e_1x1_bn_out = self.inception_4e_1x1_bn(inception_4e_1x1_out)
        inception_4e_relu_1x1_out = self.inception_4e_relu_1x1(inception_4e_1x1_bn_out)
        inception_4e_branch1_3x3_reduce_out = self.inception_4e_branch1_3x3_reduce(inception_4d_output_out)
        inception_4e_branch1_3x3_reduce_bn_out = self.inception_4e_branch1_3x3_reduce_bn(inception_4e_branch1_3x3_reduce_out)
        inception_4e_branch1_relu_3x3_reduce_out = self.inception_4e_branch1_relu_3x3_reduce(inception_4e_branch1_3x3_reduce_bn_out)
        inception_4e_branch1_3x3_out = self.inception_4e_branch1_3x3(inception_4e_branch1_3x3_reduce_bn_out)
        inception_4e_branch1_3x3_bn_out = self.inception_4e_branch1_3x3_bn(inception_4e_branch1_3x3_out)
        inception_4e_branch1_relu_3x3_out = self.inception_4e_branch1_relu_3x3(inception_4e_branch1_3x3_bn_out)
        inception_4e_branch2_3x3_reduce_out = self.inception_4e_branch2_3x3_reduce(inception_4d_output_out)
        inception_4e_branch2_3x3_reduce_bn_out = self.inception_4e_branch2_3x3_reduce_bn(inception_4e_branch2_3x3_reduce_out)
        inception_4e_branch2_relu_3x3_reduce_out = self.inception_4e_branch2_relu_3x3_reduce(inception_4e_branch2_3x3_reduce_bn_out)
        inception_4e_branch2_3x3_out = self.inception_4e_branch2_3x3(inception_4e_branch2_3x3_reduce_bn_out)
        inception_4e_branch2_3x3_bn_out = self.inception_4e_branch2_3x3_bn(inception_4e_branch2_3x3_out)
        inception_4e_branch2_relu_3x3_out = self.inception_4e_branch2_relu_3x3(inception_4e_branch2_3x3_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_pool_proj_out = self.inception_4e_pool_proj(inception_4e_pool_out)
        inception_4e_pool_proj_bn_out = self.inception_4e_pool_proj_bn(inception_4e_pool_proj_out)
        inception_4e_relu_pool_proj_out = self.inception_4e_relu_pool_proj(inception_4e_pool_proj_bn_out)
        inception_4e_output_out = torch.cat([inception_4e_1x1_bn_out,inception_4e_branch1_3x3_bn_out,inception_4e_branch2_3x3_bn_out,inception_4e_pool_proj_bn_out], 1)

        inception_4f_pool_out = self.inception_4f_pool(inception_4e_output_out)

        inception_5a_1x1_out = self.inception_5a_1x1(inception_4f_pool_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_branch1_3x3_reduce_out = self.inception_5a_branch1_3x3_reduce(inception_4f_pool_out)
        inception_5a_branch1_3x3_reduce_bn_out = self.inception_5a_branch1_3x3_reduce_bn(inception_5a_branch1_3x3_reduce_out)
        inception_5a_branch1_relu_3x3_reduce_out = self.inception_5a_branch1_relu_3x3_reduce(inception_5a_branch1_3x3_reduce_bn_out)
        inception_5a_branch1_3x3_out = self.inception_5a_branch1_3x3(inception_5a_branch1_3x3_reduce_bn_out)
        inception_5a_branch1_3x3_bn_out = self.inception_5a_branch1_3x3_bn(inception_5a_branch1_3x3_out)
        inception_5a_branch1_relu_3x3_out = self.inception_5a_branch1_relu_3x3(inception_5a_branch1_3x3_bn_out)
        inception_5a_branch2_3x3_reduce_out = self.inception_5a_branch2_3x3_reduce(inception_4f_pool_out)
        inception_5a_branch2_3x3_reduce_bn_out = self.inception_5a_branch2_3x3_reduce_bn(inception_5a_branch2_3x3_reduce_out)
        inception_5a_branch2_relu_3x3_reduce_out = self.inception_5a_branch2_relu_3x3_reduce(inception_5a_branch2_3x3_reduce_bn_out)
        inception_5a_branch2_3x3_out = self.inception_5a_branch2_3x3(inception_5a_branch2_3x3_reduce_bn_out)
        inception_5a_branch2_3x3_bn_out = self.inception_5a_branch2_3x3_bn(inception_5a_branch2_3x3_out)
        inception_5a_branch2_relu_3x3_out = self.inception_5a_branch2_relu_3x3(inception_5a_branch2_3x3_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4f_pool_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out,inception_5a_branch1_3x3_bn_out,inception_5a_branch2_3x3_bn_out,inception_5a_pool_proj_bn_out], 1)

        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_branch1_3x3_reduce_out = self.inception_5b_branch1_3x3_reduce(inception_5a_output_out)
        inception_5b_branch1_3x3_reduce_bn_out = self.inception_5b_branch1_3x3_reduce_bn(inception_5b_branch1_3x3_reduce_out)
        inception_5b_branch1_relu_3x3_reduce_out = self.inception_5b_branch1_relu_3x3_reduce(inception_5b_branch1_3x3_reduce_bn_out)
        inception_5b_branch1_3x3_out = self.inception_5b_branch1_3x3(inception_5b_branch1_3x3_reduce_bn_out)
        inception_5b_branch1_3x3_bn_out = self.inception_5b_branch1_3x3_bn(inception_5b_branch1_3x3_out)
        inception_5b_branch1_relu_3x3_out = self.inception_5b_branch1_relu_3x3(inception_5b_branch1_3x3_bn_out)
        inception_5b_branch2_3x3_reduce_out = self.inception_5b_branch2_3x3_reduce(inception_5a_output_out)
        inception_5b_branch2_3x3_reduce_bn_out = self.inception_5b_branch2_3x3_reduce_bn(inception_5b_branch2_3x3_reduce_out)
        inception_5b_branch2_relu_3x3_reduce_out = self.inception_5b_branch2_relu_3x3_reduce(inception_5b_branch2_3x3_reduce_bn_out)
        inception_5b_branch2_3x3_out = self.inception_5b_branch2_3x3(inception_5b_branch2_3x3_reduce_bn_out)
        inception_5b_branch2_3x3_bn_out = self.inception_5b_branch2_3x3_bn(inception_5b_branch2_3x3_out)
        inception_5b_branch2_relu_3x3_out = self.inception_5b_branch2_relu_3x3(inception_5b_branch2_3x3_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out,inception_5b_branch1_3x3_bn_out,inception_5b_branch2_3x3_bn_out,inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def train(self, mode=True):
        super(InceptionV1_I3D, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for n, m in self.named_modules():
                if 'conv1' not in n and isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
