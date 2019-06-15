import math
import torch
import torch.nn as nn


def charbonnier_loss(difference, mask, alpha=1, beta=1., epsilon=0.001):
    '''
    : sum( (x*beta)^2 + epsilon^2)^alpha
    '''
    if mask is not None:
        assert difference.size(0) == mask.size(0)
        assert difference.size(2) == mask.size(2)
        assert difference.size(3) == mask.size(3)
    res = torch.pow(torch.pow(difference * beta, 2) + epsilon ** 2, alpha)
    if mask is not None:
        batch_pixels = torch.sum(mask)
        return torch.sum(res * mask) / batch_pixels
    else:
        batch_pixels = torch.numel(res)
        return torch.sum(res) / batch_pixels


def SSIM_loss(img1, img2, kernel_size=8, stride=8, c1=0.00001, c2=0.00001):
    num = img1.size(0)
    channels = img1.size(1)

    kernel_h = kernel_w = kernel_size
    sigma = (kernel_w + kernel_h) / 12.
    gauss_kernel = torch.zeros((1, 1, kernel_h, kernel_w)).type(img1.type())
    for h in range(kernel_h):
        for w in range(kernel_w):
            gauss_kernel[0, 0, h, w] = math.exp(
                -(math.pow(h - kernel_h/2.0, 2) + math.pow(- kernel_w/2.0, 2))
                / (2.0 * sigma ** 2)) / (2 * 3.14159 * sigma ** 2)
    gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
    gauss_kernel = gauss_kernel.repeat(channels, 1, 1, 1)

    gauss_filter = nn.Conv2d(channels, channels, kernel_size,
                             stride=stride, padding=0,
                             groups=channels, bias=False)
    gauss_filter.weight.data = gauss_kernel
    gauss_filter.weight.requires_grad = False

    ux = gauss_filter(img1)
    uy = gauss_filter(img2)
    sx2 = gauss_filter(img1 ** 2)
    sy2 = gauss_filter(img2 ** 2)
    sxy = gauss_filter(img1 * img2)

    ux2 = ux ** 2
    uy2 = uy ** 2
    sx2 = sx2 - ux2
    sy2 = sy2 - uy2
    sxy = sxy - ux * uy

    lp = (2 * ux * uy + c1) / (ux2 + uy2 + c1)
    sc = (2 * sxy + c2) / (sx2 + sy2 + c2)

    ssim = lp * sc
    return (lp.numel() - torch.sum(ssim)) / num
