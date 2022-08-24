"""
Task: OCR

Real-time Arbitrarily-Shaped Text Spotting with Point Gathering Network
    Paper: https://arxiv.org/pdf/2104.05458.pdf
    Code: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppocr/modeling/heads/e2e_pg_head.py
"""
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones.resnet import build_resnet


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation=None):
        super(ConvBlock, self).__init__()
        try:
            self.act = getattr(F, activation)
        except (AttributeError, TypeError):
            self.act = nn.Identity()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

        
class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, groups=1, activation=None):
        super(DeConvBlock, self).__init__()
        try:
            self.act = getattr(F, activation)
        except (AttributeError, TypeError):
            self.act = nn.Identity()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, groups=groups
                                        , kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PGHead(nn.Module):

    def __init__(self, in_channels: int, num_characters: int, **kwargs):
        super(PGHead, self).__init__()
        self.num_characters = nClss = num_characters

        self.conv_center_1 = ConvBlock(in_channels=in_channels, out_channels= 64, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_center_2 = ConvBlock(in_channels=         64, out_channels= 64, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_center_3 = ConvBlock(in_channels=         64, out_channels=128, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_center   = nn.Conv2d(in_channels=        128, out_channels=  1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

        self.conv_border_1 = ConvBlock(in_channels=in_channels, out_channels= 64, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_border_2 = ConvBlock(in_channels=         64, out_channels= 64, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_border_3 = ConvBlock(in_channels=         64, out_channels=128, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_border   = nn.Conv2d(in_channels=        128, out_channels=  4, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

        self.conv_direct_1 = ConvBlock(in_channels=in_channels, out_channels= 64, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_direct_2 = ConvBlock(in_channels=         64, out_channels= 64, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_direct_3 = ConvBlock(in_channels=         64, out_channels=128, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_direct   = nn.Conv2d(in_channels=        128, out_channels=  2, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

        self.conv_char_1 = ConvBlock(in_channels=in_channels, out_channels=  128, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_char_2 = ConvBlock(in_channels=        128, out_channels=  128, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_char_3 = ConvBlock(in_channels=        128, out_channels=  256, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_char_4 = ConvBlock(in_channels=        256, out_channels=  256, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_char_5 = ConvBlock(in_channels=        256, out_channels=  256, kernel_size=1, stride=1, padding=0, activation='relu')
        self.conv_char   = nn.Conv2d(in_channels=        256, out_channels=nClss, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, targets=None):

        if isinstance(x, (tuple, list)):
            x = x[-1]

        # Text Center Line
        center = self.conv_center_1(x)          # B,  64, H, W
        center = self.conv_center_2(center)     # B,  64, H, W
        center = self.conv_center_3(center)     # B, 128, H, W
        center = self.conv_center(center)       # B,   1, H, W
        center = torch.sigmoid(center)

        # Text Border Offset
        border = self.conv_border_1(x)
        border = self.conv_border_2(border)
        border = self.conv_border_3(border)
        border = self.conv_border(border)

        # Text Direction Offset
        direction = self.conv_direct_1(x)
        direction = self.conv_direct_2(direction)
        direction = self.conv_direct_3(direction)
        direction = self.conv_direct(direction)

        # Text Character Classification
        character = self.conv_char_1(x)
        character = self.conv_char_2(character)
        character = self.conv_char_3(character)
        character = self.conv_char_4(character)
        character = self.conv_char_5(character)
        character = self.conv_char(character)

        predicts = dict()
        predicts['center'] = center
        predicts['border'] = border
        predicts['direction'] = direction
        predicts['character'] = character
        return predicts


class PGFPN(nn.Module):

    def __init__(self, **kwargs):
        super(PGFPN, self).__init__()
        self.out_channels = 128
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv_bn_layer_1 = ConvBlock(in_channels=  3, out_channels= 32, kernel_size=3, stride=1, padding=1)
        self.conv_bn_layer_2 = ConvBlock(in_channels= 64, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.conv_bn_layer_3 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_bn_layer_4 = ConvBlock(in_channels= 32, out_channels= 64, kernel_size=3, stride=2, padding=1)
        self.conv_bn_layer_5 = ConvBlock(in_channels= 64, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.conv_bn_layer_6 = ConvBlock(in_channels= 64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_bn_layer_7 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_bn_layer_8 = ConvBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)

        num_inputs = [2048, 2048, 1024, 512, 256]
        num_outputs = [256, 256, 192, 192, 128]

        self.conv_h0 =   ConvBlock(in_channels=num_inputs[0], out_channels=num_outputs[0], kernel_size=1, stride=1, padding=1)
        self.conv_h1 =   ConvBlock(in_channels=num_inputs[1], out_channels=num_outputs[1], kernel_size=1, stride=1, padding=1)
        self.conv_h2 =   ConvBlock(in_channels=num_inputs[2], out_channels=num_outputs[2], kernel_size=1, stride=1, padding=1)
        self.conv_h3 =   ConvBlock(in_channels=num_inputs[3], out_channels=num_outputs[3], kernel_size=1, stride=1, padding=1)
        self.conv_h4 =   ConvBlock(in_channels=num_inputs[4], out_channels=num_outputs[4], kernel_size=1, stride=1, padding=1)
            
        self.deconv0 = DeConvBlock(in_channels=num_outputs[0], out_channels=num_outputs[1])
        self.deconv1 = DeConvBlock(in_channels=num_outputs[1], out_channels=num_outputs[2])
        self.deconv2 = DeConvBlock(in_channels=num_outputs[2], out_channels=num_outputs[3])
        self.deconv3 = DeConvBlock(in_channels=num_outputs[3], out_channels=num_outputs[4])

        self.conv_g1 = ConvBlock(in_channels=num_outputs[1], out_channels=num_outputs[1], kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_g2 = ConvBlock(in_channels=num_outputs[2], out_channels=num_outputs[2], kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_g3 = ConvBlock(in_channels=num_outputs[3], out_channels=num_outputs[3], kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv_g4 = ConvBlock(in_channels=num_outputs[4], out_channels=num_outputs[4], kernel_size=3, stride=1, padding=1, activation='relu')
        self.convf   = ConvBlock(in_channels=num_outputs[4], out_channels=num_outputs[4], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c0, c1, c2, c3, c4, c5, c6 = x
        fd = [c0, c1, c2]
        fu = [c6, c5, c4, c3, c2]

        # FPN Down Fusion
        g = [None, None, None]
        h = [None, None, None]
        h[0] = self.conv_bn_layer_1(fd[0])
        h[1] = self.conv_bn_layer_2(fd[1])
        h[2] = self.conv_bn_layer_3(fd[2])

        g[0] = self.conv_bn_layer_4(h[0])
        g[1] = torch.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_bn_layer_5(g[1])
        g[1] = self.conv_bn_layer_6(g[1])

        g[2] = torch.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2]   = self.conv_bn_layer_7(g[2])
        f_down = self.conv_bn_layer_8(g[2])

        # FPN Up Fusion
        g = [None, None, None, None, None]
        h = [None, None, None, None, None]
        h[0] = self.conv_h0(fu[0])
        h[1] = self.conv_h1(fu[1])
        h[2] = self.conv_h2(fu[2])
        h[3] = self.conv_h3(fu[3])
        h[4] = self.conv_h4(fu[4])

        g[0] = self.deconv0(h[0])

        g[1] = torch.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_g1(g[1])
        g[1] = self.deconv1(g[1])

        g[2] = torch.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_g2(g[2])
        g[2] = self.deconv2(g[2])

        g[3] = torch.add(g[2], h[3])
        g[3] = F.relu(g[3])
        g[3] = self.conv_g3(g[3])
        g[3] = self.deconv3(g[3])

        g[4] = torch.add(x=g[3], y=h[4])
        g[4] = F.relu(g[4])
        g[4] = self.conv_g4(g[4])
        f_up = self.convf(g[4])

        f_common = torch.add(f_down, f_up)
        f_common = F.relu(f_common)
        return f_common


class PGNet(nn.Module):

    def __init__(self, in_channels: int, num_characters: int, use_fpn: bool = True, **kwargs):
        super(PGNet, self).__init__()
        self.num_characters = num_characters
        self.return_all_feats = kwargs.get("return_all_feats", False)

        if use_fpn:
            self.neck = PGFPN()
        else:
            self.neck = nn.Identity()
        self.head = PGHead(in_channels, num_characters)

    def forward(self, x):
        x = self.neck(x)
        x = self.head(x)
        return x






