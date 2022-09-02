import math

import torch
from torch import nn
from torch.nn import functional as F


class Conv2dDynamicSamePadding(nn.Conv2d):
    """
    2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.

    Tips for 'SAME' mode padding.
        Given the following:
            i: width or height
            s: stride
            k: kernel size
            d: dilation
            p: padding

        Output after Conv2d:
            o = floor( (i + p - ((k-1)*d+1)) / s + 1 )

    If o equals i, i = floor((i + p - ((k-1)*d+1))/s + 1),
    => p = (i-1)*s + ((k-1)*d + 1) - i
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride !!!
        pad_h = max((oh-1) * self.stride[0] + (kh-1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow-1) * self.stride[1] + (kw-1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


