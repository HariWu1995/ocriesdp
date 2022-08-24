import math
import torch
from torch import nn
from torch.nn import functional as F


class PRENHead(nn.Layer):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PRENHead, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, targets=None):
        predicts = self.linear(x)

        if not self.training:
            predicts = F.softmax(predicts, axis=2)

        return predicts


