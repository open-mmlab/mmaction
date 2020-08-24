import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from mmcv.runner import load_checkpoint

from ...registry import BACKBONES


__all__ = ['C3D']

@BACKBONES.register_module
class C3D(nn.Module):

    ## TODO:
    ## Refactor it into a more modular way
    ## Reference: https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_train_ucf101/conv3d_ucf101_train.prototxt

    def __init__(self,
                 pretrained=None,
                 modality='RGB'):
        super(C3D, self).__init__()

        self.pretrained = pretrained
        self.modality = modality

        inplace = True
        assert modality in ['RGB']
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu1a = nn.ReLU(inplace)
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu2a = nn.ReLU(inplace)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu3a = nn.ReLU(inplace)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu3b = nn.ReLU(inplace)
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu4a = nn.ReLU(inplace)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu4b = nn.ReLU(inplace)
        self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu5a = nn.ReLU(inplace)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.relu5b = nn.ReLU(inplace)
        self.pool5 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)

        self.fc6 = nn.Linear(8192, 4096)
        self.relu6 = nn.ReLU(inplace)
        self.drop6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace)


    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    normal_init(m, std=0.01, bias=1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.005, bias=1)


    def forward(self, input):
        conv1a = self.conv1a(input)
        conv1a = self.relu1a(conv1a)
        pool1 = self.pool1(conv1a)

        conv2a = self.conv2a(pool1)
        conv2a = self.relu2a(conv2a)
        pool2 = self.pool2(conv2a)

        conv3a = self.conv3a(pool2)
        conv3a = self.relu3a(conv3a)
        conv3b = self.conv3b(conv3a)
        conv3b = self.relu3b(conv3b)
        pool3 = self.pool3(conv3b)

        conv4a = self.conv4a(pool3)
        conv4a = self.relu4a(conv4a)
        conv4b = self.conv4b(conv4a)
        conv4b = self.relu4b(conv4b)
        pool4 = self.pool4(conv4b)

        conv5a = self.conv5a(pool4)
        conv5a = self.relu5a(conv5a)
        conv5b = self.conv5b(conv5a)
        conv5b = self.relu5b(conv5b)
        pool5 = self.pool5(conv5b)

        pool5 = pool5.flatten(start_dim=1)
        # (N, C, T, H, W) -> (N, C)
        fc6 = self.fc6(pool5)
        fc6 = self.relu6(fc6)
        fc6 = self.drop6(fc6)

        fc7 = self.fc7(fc6)
        fc7 = self.relu7(fc7)
        fc7 = fc7.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # (N, C) -> (N, C, 1, 1, 1)

        return fc7

    def train(self, mode=True):
        super(C3D, self).train(mode)
