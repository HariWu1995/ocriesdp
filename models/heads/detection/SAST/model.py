import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                                        groups=groups, stride=stride, padding=(kernel_size-1) // 2,)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SAST_Header1(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(SAST_Header1, self).__init__()
        out_channels = [64, 64, 128]
        self.score_conv = nn.Sequential(
            ConvBNLayer( in_channels   , out_channels[0], 1, 1,),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1,),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1,),
            ConvBNLayer(out_channels[2],              1 , 3, 1,)
        )
        self.border_conv = nn.Sequential(
            ConvBNLayer(in_channels    , out_channels[0], 1, 1,),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1,),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1,),
            ConvBNLayer(out_channels[2],              4 , 3, 1,)            
        )

    def forward(self, x):
        f_score = self.score_conv(x)
        f_score = F.sigmoid(f_score)
        f_border = self.border_conv(x)
        return f_score, f_border


class SAST_Header2(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(SAST_Header2, self).__init__()
        out_channels = [64, 64, 128]
        self.tvo_conv = nn.Sequential(
            ConvBNLayer( in_channels   , out_channels[0], 1, 1,),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1,),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1,),
            ConvBNLayer(out_channels[2],              8 , 3, 1,)
        )
        self.tco_conv = nn.Sequential(
            ConvBNLayer(in_channels    , out_channels[0], 1, 1,),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1,),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1,),
            ConvBNLayer(out_channels[2],              2 , 3, 1,)            
        )

    def forward(self, x):
        f_tvo = self.tvo_conv(x)
        f_tco = self.tco_conv(x)
        return f_tvo, f_tco


class SASTHead(nn.Module):
    """
    A Single-shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning
    """
    def __init__(self, in_channels, **kwargs):
        super(SASTHead, self).__init__()

        self.head1 = SAST_Header1(in_channels)
        self.head2 = SAST_Header2(in_channels)

    def forward(self, x, targets=None):
        f_score, f_border = self.head1(x)
        f_tvo  , f_tco    = self.head2(x)

        predicts = {}
        predicts['f_score' ] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts


