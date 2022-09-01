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

from models.backbones.resnet import build_resnet
from models.backbones.unet import UNet


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
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, groups=groups, 
                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
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


class PGNet(nn.Module):

    def __init__(self, in_channels: int, num_characters: int, use_fpn: bool = True, **kwargs):
        super(PGNet, self).__init__()
        self.num_characters = num_characters
        self.return_all_feats = kwargs.get("return_all_feats", False)

        if use_fpn:
            self.back = build_resnet(output_channels=2048, variant_depth=50, load_pretrained=True)
            self.neck = UNet(n_channels=256)    # 256 for H/4, W/4
        else:
            self.back = build_resnet(output_channels=512, variant_depth=50, load_pretrained=True)
            self.neck = nn.Identity()
        self.head = PGHead(in_channels, num_characters)

    def forward(self, x):
        x = self.back(dict(x=x, output='div4'))
        x = self.neck(x)
        x = self.head(x)
        return x






