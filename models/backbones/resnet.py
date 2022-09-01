'''
Clone from 
    https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/faster_rcnn/resnet.py 

with modifications:
    * remove last two layers (fc, conv)
    * modified conv1, maxpool layers
    * add conv2, bn2, relu2 layers for output final feature maps
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


pretrained_models = {
     18: 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
     34: 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
     50: 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    101: 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    152: 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

__all__ = ['ResNet',] +[f'resnet{d}' for d in pretrained_models.keys()]


def conv3x3(in_planes, out_planes, stride: int=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # change
                              
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, depth, output_channels: int=2048):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change

        self.layer1 = self._make_layer(block,  64, depth[0], stride=1)
        self.layer2 = self._make_layer(block, 128, depth[1], stride=2)
        self.layer3 = self._make_layer(block, 256, depth[2], stride=2)
        self.layer4 = self._make_layer(block, 512, depth[3], stride=2) # stride = 1: slightly better but slower

        self.conv2 = nn.Conv2d(512 * block.expansion, output_channels, kernel_size=7, stride=1, padding=1, bias=False)  # add
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m.kernel_size, (list, tuple)):
                    kernel_size = m.kernel_size
                else:
                    kernel_size = (m.kernel_size, m.kernel_size)
                n = kernel_size[0] * kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, depth, stride: int=1):
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for d in range(1, depth):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, output: str = 'all'):
        if output == 'all':
            Ys = [x]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if output == 'all':
            Ys.append(x)

        y = self.maxpool(x)

        y = self.layer1(y)
        if output in ['layer1', 'div2']:
            return y    # B, 64*expansion, H/2, W/2
        elif output == 'all':
            Ys.append(y)

        y = self.layer2(y)
        if output in ['layer2', 'div4']:
            return y    # B, 128*expansion, H/4, W/4
        elif output == 'all':
            Ys.append(y)

        y = self.layer3(y)
        if output in ['layer3', 'div8']:
            return y    # B, 256*expansion, H/8, W/8
        elif output == 'all':
            Ys.append(y)

        y = self.layer4(y)
        if output in ['layer4', 'div16']:
            return y    # B, 512*expansion, H/16, W/16
        elif output == 'all':
            Ys.append(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        if output == 'all':
            Ys.append(y)
            return Ys

        return y    # B, 512, H/16, W/16


model_architecture = {
     18: (BasicBlock, [2, 2,  2, 2], ),
     34: (BasicBlock, [3, 4,  6, 3], ),
     50: (Bottleneck, [3, 4,  6, 3], ),
    101: (Bottleneck, [3, 4, 23, 3], ),
    152: (Bottleneck, [3, 8, 36, 3], ),
}

def build_resnet(output_channels: int=512, variant_depth: int=50, load_pretrained: bool=False):
    if variant_depth not in model_architecture.keys():
        variant_depth = 50
    block, depth = model_architecture.get(variant_depth, (Bottleneck, [3, 4,  6, 3], ))
    model = ResNet(block, depth, output_channels)
    if load_pretrained and (variant_depth in pretrained_models.keys()):
        model.load_state_dict(model_zoo.load_url(pretrained_models[variant_depth]), strict=False)
    return model




