from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from src.models.activations import HardSigmoid, HardSwish


__all__ = ['MobileNetV3', 'mobilenetv3']


def make_divisible(x, divisible_by: int=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_bn(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, activation=None):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU() if not isinstance(activation, nn.Module) else activation
    )


class SqueezeExcite(nn.Module):

    def __init__(self, channel, reduction: int = 4):
        super(SqueezeExcite, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid(inplace=True)
            # nn.Sigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int, 
                    expansion_size: int, use_SE: bool = False, activation: str = 'RE'):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        padding = (kernel_size - 1) // 2
        self.use_residual = stride == 1 and in_planes == out_planes

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        activation = activation.upper()
        if activation == 'RE':
            activation = nn.ReLU # or ReLU6
        elif activation == 'HS':
            activation = HardSwish
        else:
            raise NotImplementedError

        if use_SE:
            se_layer = SqueezeExcite
        else:
            se_layer = nn.Identity

        self.conv = nn.Sequential(
            # Point-wise
            conv_layer(in_planes, expansion_size, 1, 1, 0, bias=False),
            norm_layer(expansion_size),
            activation(inplace=True),

            # Depth-wise
            conv_layer(expansion_size, expansion_size, kernel_size, stride, padding, groups=expansion_size, bias=False),
            norm_layer(expansion_size),
              se_layer(expansion_size),
            activation(inplace=True),
            
            # pw-linear
            conv_layer(expansion_size, out_planes, 1, 1, 0, bias=False),
            norm_layer(out_planes),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):

    def __init__(self, input_size: int = 224, mode: str = 'small', width_mult: float = 1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280

        if mode == 'large':
            # refer to Table 1 in paper
            model_architecture = [
                # k, exp, c,  se,     act, s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]

        elif mode == 'small':
            # refer to Table 2 in paper
            model_architecture = [
                # k, exp, c,  se,    act, s,
                [3, 16,  16, True,  'RE', 2],
                [3, 72,  24, False, 'RE', 2],
                [3, 88,  24, False, 'RE', 1],
                [5, 96,  40, True,  'HS', 2],
                [5, 240, 40, True,  'HS', 1],
                [5, 240, 40, True,  'HS', 1],
                [5, 120, 48, True,  'HS', 1],
                [5, 144, 48, True,  'HS', 1],
                [5, 288, 96, True,  'HS', 2],
                [5, 576, 96, True,  'HS', 1],
                [5, 576, 96, True,  'HS', 1],
            ]

        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0, f"Require input_size being divided by 32, while image has size {input_size}"
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.layers = [conv_bn(3, input_channel, kernel_size=3, stride=2, padding=1, activation=HardSwish)]

        # building mobile blocks
        for k, exp, c, se, act, s in model_architecture:
            output_channel = make_divisible(c * width_mult)
            expand_channel = make_divisible(exp * width_mult)
            self.layers.append(Bottleneck(input_channel, output_channel, k, s, expand_channel, se, act))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.layers.append(conv_bn(input_channel, last_conv, kernel_size=1, stride=1, padding=0, activation=HardSwish))
            self.layers.append(nn.AdaptiveAvgPool2d(1))
            self.layers.append(nn.Conv2d(last_conv, last_channel, kernel_size=1, stride=1, padding=0,))
            self.layers.append(HardSwish(inplace=True))

        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.layers.append(conv_bn(input_channel, last_conv, kernel_size=1, stride=1, padding=0, activation=HardSwish))
            # self.layers.append(SqueezeExcite(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.layers.append(nn.AdaptiveAvgPool2d(1))
            self.layers.append(nn.Conv2d(last_conv, last_channel, kernel_size=1, stride=1, padding=0,))
            self.layers.append(HardSwish(inplace=True))

        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.layers = nn.Sequential(*self.layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        x = x.mean(3).mean(2)
        return x


def build_mobilenetv3(input_size, variant: str = 'small', width_mult: float = 1.0, pretrained_path: str = None):
    model = MobileNetV3(input_size, mode=variant, width_mult=width_mult)
    if pretrained_path is not None:
        if os.path.isfile(pretrained_path):
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)
    return model







