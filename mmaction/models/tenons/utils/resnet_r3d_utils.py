"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import torch.nn as nn
import string
import itertools


def conv3d_wobias(in_planes, out_planes, kernel, stride, pad, groups=1):
    assert len(kernel) == 3
    assert len(stride) == 3
    assert len(pad) == 3
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel,
                     stride=stride, padding=pad, groups=groups, bias=False)

def conv3d_wbias(in_planes, out_planes, kernel, stride, pad, groups=1):
    assert len(kernel) == 3
    assert len(stride) == 3
    assert len(pad) == 3
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel,
                     stride=stride, padding=pad, groups=groups, bias=True)


class module_list(nn.Module):
    def __init__(self, modules, names=None):
        super(module_list, self).__init__()
        self.num = len(modules)
        self.modules = modules
        if names is None:
            alphabet = string.ascii_lowercase
            alphabet = list(alphabet)
            if self.num < 26:
                self.names = alphabet[:self.num]
            else:
                alphabet2 = itertools.product(alphabet, alphabet)
                alphabet2 = list(map(lambda x: x[0] + x[1], alphabet2))
                self.names = alphabet2[:self.num]
        else:
            assert len(names) == self.num
            self.names = names
        for m, n in zip(self.modules, self.names):
            setattr(self, n, m)

    def forward(self, inp):
        for n in self.names:
            inp = getattr(self, n)(inp)
        return inp


def add_conv3d(in_filters, out_filters, kernel, stride, pad, block_type='3d', group=1, with_bn=True):
    if with_bn:
        conv3d = conv3d_wobias
    else:
        conv3d = conv3d_wbias
    if block_type == '2.5d':
        i = 3 * in_filters * out_filters * kernel[1] * kernel[2]
        i /= in_filters * kernel[1] * kernel[2] + 3 * out_filters
        middle_filters = int(i)
        conv_s = conv3d(in_filters, middle_filters, kernel=[1, kernel[1], kernel[2]],
                        stride=[1, stride[1], stride[2]], pad=[0, pad[1], pad[2]])
        bn_s = nn.BatchNorm3d(middle_filters, eps=1e-3)
        conv_t = conv3d(middle_filters, out_filters, kernel=[kernel[0], 1, 1],
                        stride=[stride[0], 1, 1], pad=[pad[0], 0, 0])
        if with_bn:
            return module_list([conv_s, bn_s, nn.ReLU(), conv_t], ['conv_s', 'bn_s', 'relu_s', 'conv_t'])
        else:
            return module_list([conv_s, nn.ReLU(), conv_t], ['conv_s', 'relu_s', 'conv_t'])
    if block_type == '0.3d':
        conv_T = conv3d(in_filters, out_filters, kernel=[
                        1, 1, 1], stride=[1, 1, 1], pad=[0, 0, 0])
        bn_T = nn.BatchNorm3d(out_filters, eps=1e-3)
        conv_C = conv3d(out_filters, out_filters,
                        kernel=kernel, stride=stride, pad=pad)
        if with_bn:
            return module_list([conv_T, bn_T, nn.ReLU(), conv_C], ['conv_T', 'bn_T', 'relu_T', 'conv_C'])
        else:
            return module_list([conv_T, nn.ReLU(), conv_C], ['conv_T', 'relu_T', 'conv_C'])
    if block_type == '3d':
        conv = conv3d(in_filters, out_filters,
                      kernel=kernel, stride=stride, pad=pad)
        return conv
    if block_type == '3d-sep':
        assert in_filters == out_filters
        conv = conv3d(in_filters, out_filters, kernel=kernel,
                      stride=stride, pad=pad, groups=in_filters)
        return conv
    print('Unknown Block Type !!!')


def add_bn(num_filters):
    bn = nn.BatchNorm3d(num_filters, eps=1e-3)
    return bn
