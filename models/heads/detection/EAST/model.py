import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, groups=groups,)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class EASTHead(nn.Module):
    """
    EAST: An Efficient and Accurate Scene Text Detector
    """
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTHead, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [ 64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(in_channels=in_channels, out_channels=num_outputs[0],
                                    kernel_size=3, stride=1, padding=1,)
        self.det_conv2 = ConvBNLayer(in_channels=num_outputs[0], out_channels=num_outputs[1],
                                    kernel_size=3, stride=1, padding=1,)
        self.score_conv = ConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[2],
                                    kernel_size=1, stride=1, padding=0,)
        self.geo_conv = ConvBNLayer(in_channels=num_outputs[1], out_channels=num_outputs[3],
                                    kernel_size=1, stride=1, padding=0,)

    def forward(self, x, targets=None):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = F.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        return {
            'f_score': f_score, 
            'f_geo': f_geo,
        }


