import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....ops.resample2d_package.resample2d import Resample2d
from ....losses import charbonnier_loss, SSIM_loss
from mmcv.cnn import kaiming_init
from mmcv.runner import load_checkpoint

from ...registry import FLOWNETS

def make_smoothness_mask(batch, height, width, tensor_type):
    mask = torch.ones(batch, 2, height, width).type(tensor_type)
    mask[: 1, -1, :] = 0
    mask[: 0, :, -1] = 0
    return mask

def make_border_mask(batch, channels, height, width, tensor_type, border_ratio=0.1):
    border_width = round(border_ratio * min(height, width))
    mask = torch.ones(batch, channels, height, width).type(tensor_type)
    mask[:, :, :border_width, :] = 0
    mask[:, :, -border_width:, :] = 0
    mask[:, :, :border_width, :] = 0
    mask[:, :, -border_width:, :] = 0
    return mask

@FLOWNETS.register_module
class MotionNet(nn.Module):


    def __init__(self,
                 num_frames=1,
                 rgb_disorder=False,
                 scale=0.0039216,
                 out_loss_indices=(0, 1, 2, 3, 4),
                 out_prediction_indices=(0, 1, 2, 3, 4),
                 out_prediction_rescale=True,
                 frozen=False,
                 use_photometric_loss=True,
                 use_ssim_loss=True,
                 use_smoothness_loss=True,
                 photometric_loss_weights=(1, 1, 1, 1, 1),
                 ssim_loss_weights=(0.16, 0.08, 0.04, 0.02, 0.01),
                 smoothness_loss_weights=(1, 1, 1, 1, 1),
                 pretrained=None):
        super(MotionNet, self).__init__()
        self.num_frames = num_frames
        self.rgb_disorder = rgb_disorder
        self.scale = scale
        self.out_loss_indices = out_loss_indices
        self.out_prediction_indices = out_prediction_indices
        self.out_prediction_rescale = out_prediction_rescale
        self.use_photometric_loss = use_photometric_loss
        self.use_ssim_loss = use_ssim_loss
        self.use_smoothness_loss = use_smoothness_loss
        if frozen:
            self.use_photometric_loss = False
            self.use_ssim_loss = False
            self.use_smoothness_loss = False
        self.frozen = frozen
        self.photometric_loss_weights = photometric_loss_weights
        self.ssim_loss_weights = ssim_loss_weights
        self.smoothness_loss_weights = smoothness_loss_weights
        if use_photometric_loss:
            assert(len(out_prediction_indices) == len(photometric_loss_weights))
        if use_ssim_loss:
            assert(len(out_prediction_indices) == len(ssim_loss_weights))
        if use_smoothness_loss:
            assert(len(out_prediction_indices) == len(smoothness_loss_weights))
        self.pretrained=pretrained
        inplace = True
        self.lrn = nn.LocalResponseNorm(9, alpha=1, beta=0.5)  # norm images for photometric losses
        self.conv1 = nn.Conv2d(3*(num_frames+1), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu1_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu2_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu3_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu4_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu5_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        #### bottleneck layer ####
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu6_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        
        self.conv_pr6 = nn.Conv2d(1024, 2*num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        
        self.warp6 = Resample2d()
        self.warp5 = Resample2d()
        self.warp4 = Resample2d()
        self.warp3 = Resample2d()
        self.warp2 = Resample2d()

        #### FlowDelta filter (fixed) ####
        self.conv_FlowDelta = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_FlowDelta.weight = nn.Parameter(torch.Tensor([[[[0,0,0],[0,1,-1],[0,0,0]]],[[[0,0,0],[0,1,0],[0,-1,0]]]]))
        self.conv_FlowDelta.weight.requires_grad = False
        
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up5 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow6to5 = nn.ConvTranspose2d(2*self.num_frames, 2*self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv5 = nn.Conv2d(2*(self.num_frames+512), 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr5 = nn.Conv2d(512, 2*num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up4 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2*self.num_frames, 2*self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv4 = nn.Conv2d((2*self.num_frames+512+256), 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr4 = nn.Conv2d(256, 2*num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up3 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2*self.num_frames, 2*self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv3 = nn.Conv2d((2*self.num_frames+256+128), 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr3 = nn.Conv2d(128, 2*num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2*self.num_frames, 2*self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv2 = nn.Conv2d((2*self.num_frames+128+64), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr2 = nn.Conv2d(64, 2*num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

        self.init_weights()

    @property
    def flip_rgb(self):
        return self.rgb_disorder

    @property
    def multiframe(self):
        return self.num_frames > 1

    def forward(self, x, train=True):
        assert(x.ndimension() == 4)
        scaling = torch.tensor(self.scale * (self.num_frames + 1)).type(x.type()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = x * scaling
        imgs = torch.split(x, 3, 1)
        assert(len(imgs) == self.num_frames + 1)
        imgs_norm = [self.lrn(imgs[i]) for i in range(self.num_frames + 1)]

        conv1 = self.conv1(x)
        conv1_relu = self.relu1(conv1)
        conv1_1 = self.conv1_1(conv1_relu)
        conv1_1_relu = self.relu1_1(conv1_1)
        conv2 = self.conv2(conv1_1_relu)
        conv2_relu = self.relu2(conv2)
        conv2_1 = self.conv2_1(conv2_relu)
        conv2_1_relu = self.relu2_1(conv2_1)
        conv3 = self.conv3(conv2_1_relu)
        conv3_relu = self.relu3(conv3)
        conv3_1 = self.conv3_1(conv3_relu)
        conv3_1_relu = self.relu3_1(conv3_1)
        conv4 = self.conv4(conv3_1_relu)
        conv4_relu = self.relu4(conv4)
        conv4_1 = self.conv4_1(conv4_relu)
        conv4_1_relu = self.relu4_1(conv4_1)
        conv5 = self.conv5(conv4_1_relu)
        conv5_relu = self.relu5(conv5)
        conv5_1 = self.conv5_1(conv5_relu)
        conv5_1_relu = self.relu5_1(conv5_1)

        conv6 = self.conv6(conv5_1_relu)
        conv6_relu = self.relu6(conv6)
        conv6_1 = self.conv6_1(conv6_relu)
        conv6_1_relu = self.relu6_1(conv6_1)
        predict_flow6 = self.conv_pr6(conv6_1_relu)

        predictions_outs = []
        photometric_loss_outs = []
        ssim_loss_outs = []
        smoothness_loss_outs = []
        FlowScale6 = predict_flow6 * 0.625
        
        if train:
            #### for loss 6 ####
            predict_flow6_xs = torch.split(predict_flow6, 2, 1)
            FlowScale6_xs = torch.split(FlowScale6, 2, 1)
            downsampled_imgs_6 = [F.interpolate(img_norm, size=predict_flow6_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            #### warp img1 to back img0 ####
            #### for photometric loss ####
            #### for SSIM loss ####
            Warped6_xs = [self.warp6(downsampled_imgs_6[i+1].contiguous(), FlowScale6_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled6_input_concat = torch.cat(downsampled_imgs_6[: self.num_frames], 1)
            warped6_concat = torch.cat(Warped6_xs, 1)
            PhotoDifference6 = downsampled6_input_concat - warped6_concat
            #### flow gradients ####
            #### for smoothness loss ####
            U6 = predict_flow6[:, ::2, ...]
            V6 = predict_flow6[:, 1::2, ...]
            FlowDeltasU6 = self.conv_FlowDelta(U6.view(-1, 1, U6.size(2), U6.size(3))).view(-1, self.num_frames, 2, U6.size(2), U6.size(3))
            FlowDeltasU6_xs = torch.split(FlowDeltasU6, 1, 1)
            FlowDeltasV6 = self.conv_FlowDelta(V6.view(-1, 1, V6.size(2), V6.size(3))).view(-1, self.num_frames, 2, V6.size(2), V6.size(3))
            FlowDeltasV6_xs = torch.split(FlowDeltasV6, 1, 1)

            SmoothnessMask6 = make_smoothness_mask(U6.size(0), U6.size(2), U6.size(3), U6.type())
            FlowDeltasUClean6_xs = [FlowDeltasU6_x.squeeze() * SmoothnessMask6 for FlowDeltasU6_x in FlowDeltasU6_xs]
            FlowDeltasVClean6_xs = [FlowDeltasV6_x.squeeze() * SmoothnessMask6 for FlowDeltasV6_x in FlowDeltasV6_xs]
            FlowDeltasUClean6 = torch.cat(FlowDeltasUClean6_xs, 1)
            FlowDeltasVClean6 = torch.cat(FlowDeltasVClean6_xs, 1)

            BorderMask6 = make_border_mask(U6.size(0), 3*U6.size(1), U6.size(2), U6.size(3), U6.type(), border_ratio=0.1)

            photometric_loss_outs.append((PhotoDifference6, BorderMask6))
            ssim_loss_outs.append((warped6_concat, downsampled6_input_concat))
            BorderMask6 = make_border_mask(U6.size(0), 2*U6.size(1), U6.size(2), U6.size(3), U6.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean6, FlowDeltasVClean6, BorderMask6))
            #### loss 6 ends here ####
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale6)
        else:
            predictions_outs.append(predict_flow6)

        deconv5 = self.deconv5(conv6_1_relu)
        deconv5_relu = self.relu_up5(deconv5)
        upsampled_flow6_to_5 = self.upsample_flow6to5(predict_flow6)
        concat5 = torch.cat((conv5_1_relu, deconv5_relu, upsampled_flow6_to_5), 1)
        smooth_conv5 = self.smooth_conv5(concat5)
        predict_flow5 = self.conv_pr5(smooth_conv5)
        FlowScale5 = predict_flow5 * 1.25
        
        if train:
            #### for loss 5 ####
            predict_flow5_xs = torch.split(predict_flow5, 2, 1)
            FlowScale5_xs = torch.split(FlowScale5, 2, 1) 
            downsampled_imgs_5 = [F.interpolate(img_norm, size=predict_flow5_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            #### warp img1 to back img0 ####
            #### for photometric loss ####
            #### for SSIM loss ####
            Warped5_xs = [self.warp5(downsampled_imgs_5[i+1].contiguous(), FlowScale5_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled5_input_concat = torch.cat(downsampled_imgs_5[: self.num_frames], 1)
            warped5_concat = torch.cat(Warped5_xs, 1)
            PhotoDifference5 = downsampled5_input_concat - warped5_concat
            #### flow gradients ####
            #### for smoothness loss ####
            U5 = predict_flow5[:, ::2, ...]
            V5 = predict_flow5[:, 1::2, ...]
            FlowDeltasU5 = self.conv_FlowDelta(U5.view(-1, 1, U5.size(2), U5.size(3))).view(-1, self.num_frames, 2, U5.size(2), U5.size(3))
            FlowDeltasU5_xs = torch.split(FlowDeltasU5, 1, 1)
            FlowDeltasV5 = self.conv_FlowDelta(V5.view(-1, 1, V5.size(2), V5.size(3))).view(-1, self.num_frames, 2, V5.size(2), V5.size(3))
            FlowDeltasV5_xs = torch.split(FlowDeltasV5, 1, 1)

            SmoothnessMask5 = make_smoothness_mask(U5.size(0), U5.size(2), U5.size(3), U5.type())
            FlowDeltasUClean5_xs = [FlowDeltasU5_x.squeeze() * SmoothnessMask5 for FlowDeltasU5_x in FlowDeltasU5_xs]
            FlowDeltasVClean5_xs = [FlowDeltasV5_x.squeeze() * SmoothnessMask5 for FlowDeltasV5_x in FlowDeltasV5_xs]
            FlowDeltasUClean5 = torch.cat(FlowDeltasUClean5_xs, 1)
            FlowDeltasVClean5 = torch.cat(FlowDeltasVClean5_xs, 1)

            BorderMask5 = make_border_mask(U5.size(0), 3*U5.size(1), U5.size(2), U5.size(3), U5.type(), border_ratio=0.1)

            photometric_loss_outs.append((PhotoDifference5, BorderMask5))
            ssim_loss_outs.append((warped5_concat, downsampled5_input_concat))
            BorderMask5 = make_border_mask(U5.size(0), 2*U5.size(1), U5.size(2), U5.size(3), U5.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean5, FlowDeltasVClean5, BorderMask5))
            #### loss 5 ends here ####
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale5)
        else:
            predictions_outs.append(predict_flow5)

        deconv4 = self.deconv4(smooth_conv5)
        deconv4_relu = self.relu_up4(deconv4)
        upsampled_flow5_to_4 = self.upsample_flow5to4(predict_flow5)
        concat4 = torch.cat((conv4_1_relu, deconv4_relu, upsampled_flow5_to_4), 1)
        smooth_conv4 = self.smooth_conv4(concat4)
        predict_flow4 = self.conv_pr4(smooth_conv4)
        FlowScale4 = predict_flow4 * 2.5

        if train:
            #### for loss 4 ####
            predict_flow4_xs = torch.split(predict_flow4, 2, 1)
            FlowScale4_xs = torch.split(FlowScale4, 2, 1)
            downsampled_imgs_4 = [F.interpolate(img_norm, size=predict_flow4_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            #### warp img1 to back img0 ####
            #### for photometric loss ####
            #### for SSIM loss ####
            Warped4_xs = [self.warp4(downsampled_imgs_4[i+1].contiguous(), FlowScale4_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled4_input_concat = torch.cat(downsampled_imgs_4[: self.num_frames], 1)
            warped4_concat = torch.cat(Warped4_xs, 1)
            PhotoDifference4 = downsampled4_input_concat - warped4_concat
            #### flow gradients ####
            #### for smoothness loss ####
            U4 = predict_flow4[:, ::2, ...]
            V4 = predict_flow4[:, 1::2, ...]
            FlowDeltasU4 = self.conv_FlowDelta(U4.view(-1, 1, U4.size(2), U4.size(3))).view(-1, self.num_frames, 2, U4.size(2), U4.size(3))
            FlowDeltasU4_xs = torch.split(FlowDeltasU4, 1, 1)
            FlowDeltasV4 = self.conv_FlowDelta(V4.view(-1, 1, V4.size(2), V4.size(3))).view(-1, self.num_frames, 2, V4.size(2), V4.size(3))
            FlowDeltasV4_xs = torch.split(FlowDeltasV4, 1, 1)

            SmoothnessMask4 = make_smoothness_mask(U4.size(0), U4.size(2), U4.size(3), U4.type())
            FlowDeltasUClean4_xs = [FlowDeltasU4_x.squeeze() * SmoothnessMask4 for FlowDeltasU4_x in FlowDeltasU4_xs]
            FlowDeltasVClean4_xs = [FlowDeltasV4_x.squeeze() * SmoothnessMask4 for FlowDeltasV4_x in FlowDeltasV4_xs]
            FlowDeltasUClean4 = torch.cat(FlowDeltasUClean4_xs, 1)
            FlowDeltasVClean4 = torch.cat(FlowDeltasVClean4_xs, 1)

            BorderMask4 = make_border_mask(U4.size(0), 3*U4.size(1), U4.size(2), U4.size(3), U4.type(), border_ratio=0.1)

            photometric_loss_outs.append((PhotoDifference4, BorderMask4))
            ssim_loss_outs.append((warped4_concat, downsampled4_input_concat))
            BorderMask4 = make_border_mask(U4.size(0), 2*U4.size(1), U4.size(2), U4.size(3), U4.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean4, FlowDeltasVClean4, BorderMask4))
            #### loss 4 ends here ####
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale4)
        else:
            predictions_outs.append(predict_flow4)

        deconv3 = self.deconv3(smooth_conv4)
        deconv3_relu = self.relu_up3(deconv3)
        upsampled_flow4_to_3 = self.upsample_flow4to3(predict_flow4)
        concat3 = torch.cat((conv3_1_relu, deconv3_relu, upsampled_flow4_to_3), 1)
        smooth_conv3 = self.smooth_conv3(concat3)
        predict_flow3 = self.conv_pr3(smooth_conv3)
        FlowScale3 = predict_flow3 * 5.0

        if train:
            #### for loss 3 ####
            predict_flow3_xs = torch.split(predict_flow3, 2, 1)
            FlowScale3_xs = torch.split(FlowScale3, 2, 1) 
            downsampled_imgs_3 = [F.interpolate(img_norm, size=predict_flow3_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            #### warp img1 to back img0 ####
            #### for photometric loss ####
            #### for SSIM loss ####
            Warped3_xs = [self.warp3(downsampled_imgs_3[i+1].contiguous(), FlowScale3_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled3_input_concat = torch.cat(downsampled_imgs_3[: self.num_frames], 1)
            warped3_concat = torch.cat(Warped3_xs, 1)
            PhotoDifference3 = downsampled3_input_concat - warped3_concat
            #### flow gradients ####
            #### for smoothness loss ####
            U3 = predict_flow3[:, ::2, ...]
            V3 = predict_flow3[:, 1::2, ...]
            FlowDeltasU3 = self.conv_FlowDelta(U3.view(-1, 1, U3.size(2), U3.size(3))).view(-1, self.num_frames, 2, U3.size(2), U3.size(3))
            FlowDeltasU3_xs = torch.split(FlowDeltasU3, 1, 1)
            FlowDeltasV3 = self.conv_FlowDelta(V3.view(-1, 1, V3.size(2), V3.size(3))).view(-1, self.num_frames, 2, V3.size(2), V3.size(3))
            FlowDeltasV3_xs = torch.split(FlowDeltasV3, 1, 1)

            SmoothnessMask3 = make_smoothness_mask(U3.size(0), U3.size(2), U3.size(3), U3.type())
            FlowDeltasUClean3_xs = [FlowDeltasU3_x.squeeze() * SmoothnessMask3 for FlowDeltasU3_x in FlowDeltasU3_xs]
            FlowDeltasVClean3_xs = [FlowDeltasV3_x.squeeze() * SmoothnessMask3 for FlowDeltasV3_x in FlowDeltasV3_xs]
            FlowDeltasUClean3 = torch.cat(FlowDeltasUClean3_xs, 1)
            FlowDeltasVClean3 = torch.cat(FlowDeltasVClean3_xs, 1)

            BorderMask3 = make_border_mask(U3.size(0), 3*U3.size(1), U3.size(2), U3.size(3), U3.type(), border_ratio=0.1)

            photometric_loss_outs.append((PhotoDifference3, BorderMask3))
            ssim_loss_outs.append((warped3_concat, downsampled3_input_concat))
            BorderMask3 = make_border_mask(U3.size(0), 2*U3.size(1), U3.size(2), U3.size(3), U3.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean3, FlowDeltasVClean3, BorderMask3))
            #### loss 3 ends here ####
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale3)
        else:
            predictions_outs.append(predict_flow3)

        deconv2 = self.deconv2(smooth_conv3)
        deconv2_relu = self.relu_up2(deconv2)
        upsampled_flow3_to_2 = self.upsample_flow3to2(predict_flow3)
        concat2 = torch.cat((conv2_1_relu, deconv2_relu, upsampled_flow3_to_2), 1)
        smooth_conv2 = self.smooth_conv2(concat2)
        predict_flow2 = self.conv_pr2(smooth_conv2)
        FlowScale2 = predict_flow2 * 10.0

        if train:
            #### for loss 2 ####
            predict_flow2_xs = torch.split(predict_flow2, 2, 1)
            FlowScale2_xs = torch.split(FlowScale2, 2, 1) 
            downsampled_imgs_2 = [F.interpolate(img_norm, size=predict_flow2_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            #### warp img1 to back img0 ####
            #### for photometric loss ####
            #### for SSIM loss ####
            Warped2_xs = [self.warp2(downsampled_imgs_2[i+1].contiguous(), FlowScale2_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled2_input_concat = torch.cat(downsampled_imgs_2[: self.num_frames], 1)
            warped2_concat = torch.cat(Warped2_xs, 1)
            PhotoDifference2 = downsampled2_input_concat - warped2_concat
            #### flow gradients ####
            #### for smoothness loss ####
            U2 = predict_flow2[:, ::2, ...]
            V2 = predict_flow2[:, 1::2, ...]
            FlowDeltasU2 = self.conv_FlowDelta(U2.view(-1, 1, U2.size(2), U2.size(3))).view(-1, self.num_frames, 2, U2.size(2), U2.size(3))
            FlowDeltasU2_xs = torch.split(FlowDeltasU2, 1, 1)
            FlowDeltasV2 = self.conv_FlowDelta(V2.view(-1, 1, V2.size(2), V2.size(3))).view(-1, self.num_frames, 2, V2.size(2), V2.size(3))
            FlowDeltasV2_xs = torch.split(FlowDeltasV2, 1, 1)

            SmoothnessMask2 = make_smoothness_mask(U2.size(0), U2.size(2), U2.size(3), U2.type())
            FlowDeltasUClean2_xs = [FlowDeltasU2_x.squeeze() * SmoothnessMask2 for FlowDeltasU2_x in FlowDeltasU2_xs]
            FlowDeltasVClean2_xs = [FlowDeltasV2_x.squeeze() * SmoothnessMask2 for FlowDeltasV2_x in FlowDeltasV2_xs]
            FlowDeltasUClean2 = torch.cat(FlowDeltasUClean2_xs, 1)
            FlowDeltasVClean2 = torch.cat(FlowDeltasVClean2_xs, 1)

            BorderMask2 = make_border_mask(U2.size(0), 3*U2.size(1), U2.size(2), U2.size(3), U2.type(), border_ratio=0.1)

            if self.out_prediction_rescale:
                predictions_outs.append(FlowScale2)
            else:
                predictions_outs.append(predict_flow2)
            photometric_loss_outs.append((PhotoDifference2, BorderMask2))
            ssim_loss_outs.append((warped2_concat, downsampled2_input_concat))
            BorderMask2 = make_border_mask(U2.size(0), 2*U2.size(1), U2.size(2), U2.size(3), U2.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean2, FlowDeltasVClean2, BorderMask2))
            #### loss 2 ends here ####


        outs_predictions = [predictions_outs[i] for i in self.out_prediction_indices]
        
        if train:
            return tuple(outs_predictions), photometric_loss_outs, ssim_loss_outs, smoothness_loss_outs
        else:
            return tuple(outs_predictions), None, None, None

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    if name != "conv_FlowDelta":
                        kaiming_init(m)
                    else:
                        print("Fixing conv_FlowDelta")
        else:
            raise TypeError('pretrained must be a str or None')

    def loss(self,
             photometric_loss_outs,
             ssim_loss_outs,
             smoothness_loss_outs,
             direction='forward'):
        assert direction in ['forward', 'backward']
        losses = dict()
        # outs_photometric = dict()
        # outs_ssim = dict()
        # outs_smoothness = dict()
        if self.use_photometric_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['photometric_loss_{}_{}'.format(ind, direction)] = self.photometric_loss_weights[i] * charbonnier_loss(photometric_loss_outs[i][0], photometric_loss_outs[i][1], alpha=0.4, beta=255)
        if self.use_ssim_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['ssim_loss_{}_{}'.format(ind, direction)] = self.ssim_loss_weights[i] * SSIM_loss(ssim_loss_outs[i][0], ssim_loss_outs[i][1], kernel_size=8, stride=8, c1=0.0001, c2=0.001)
        if self.use_smoothness_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['smoothness_loss_{}_{}'.format(ind, direction)] = self.smoothness_loss_weights[i] * charbonnier_loss(smoothness_loss_outs[i][0], smoothness_loss_outs[i][2], alpha=0.3, beta=5) + \
                                                           self.smoothness_loss_weights[i] * charbonnier_loss(smoothness_loss_outs[i][1], smoothness_loss_outs[i][2], alpha=0.3, beta=5)
        return losses 


    def train(self, mode=True):
        super(MotionNet, self).train(mode)
        if self.frozen:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    for param in m.parameters():
                        param.requires_grad = False
