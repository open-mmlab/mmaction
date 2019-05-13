import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _triple

import math

import traj_conv_cuda

class TrajConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                deformable_groups=1,
                im2col_step=64):
        if input is not None and input.dim() != 5:
            raise ValueError(
                "Expected 5D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight, bias)

        output = input.new(*TrajConvFunction._output_size(
            input, weight, ctx.padding, ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new(), input.new()]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not (isinstance(input.data, torch.cuda.FloatTensor) or isinstance(input.data, torch.cuda.DoubleTensor)):
                    raise NotImplementedError
            else:
                if not (isinstance(input, torch.cuda.FloatTensor) or isinstance(input, torch.cuda.DoubleTensor)):
                    raise NotImplementedError

            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            traj_conv_cuda.deform_3d_conv_forward_cuda(
                input, weight, bias, offset, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(2), weight.size(3), weight.size(4),
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2], ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = grad_bias = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not (isinstance(grad_output.data, torch.cuda.FloatTensor) or isinstance(grad_output.data, torch.cuda.DoubleTensor)):
                    raise NotImplementedError
            else:
                if not (isinstance(grad_output, torch.cuda.FloatTensor) or isinstance(grad_output, torch.cuda.DoubleTensor)):
                    raise NotImplementedError

            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                # print("input.size: ", input.size())
                # print("offset.size: ", offset.size())
                # print("grad_output.size: ", grad_output.size())
                # print("grad_input.size: ", grad_input.size())
                # print("grad_offset.size: ", grad_offset.size())
                traj_conv_cuda.deform_3d_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, bias, ctx.bufs_[0],
                    weight.size(2), weight.size(3), weight.size(4),
                    ctx.stride[0], ctx.stride[1], ctx.stride[2],
                    ctx.padding[0], ctx.padding[1], ctx.padding[2],
                    ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                    ctx.deformable_groups, cur_im2col_step)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                grad_bias = torch.zeros_like(bias)
                traj_conv_cuda.deform_3d_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, grad_bias, ctx.bufs_[0], ctx.bufs_[1], 
                    weight.size(2), weight.size(3), weight.size(4),
                    ctx.stride[0], ctx.stride[1], ctx.stride[2],
                    ctx.padding[0], ctx.padding[1], ctx.padding[2],
                    ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                    ctx.deformable_groups, 1, cur_im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None, None


    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class TrajConv(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1,
                 im2col_step=64,
                 bias=True):
        super(TrajConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.num_deformable_groups = num_deformable_groups
        self.im2col_step = im2col_step

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_channels,))
        else:
            self.bias = nn.Parameter(
                torch.zeros(0,))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias.nelement() != 0:
            self.bias.data.fill_(0.)

    def forward(self, input, offset):
        return TrajConvFunction.apply(input, offset, self.weight, self.bias, self.stride,
                                      self.padding, self.dilation,
                                      self.num_deformable_groups,
                                      self.im2col_step)
