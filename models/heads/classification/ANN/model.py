import math
import torch
from torch import nn
from torch.nn import functional as F


class ClassfierHead(nn.Module):

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClassfierHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        stdv = 1.0 / math.sqrt(in_channels * 1.0)
        self.fc = nn.Linear(in_channels, class_dim,)

    def forward(self, x, targets=None):
        x = self.pool(x)
        x = torch.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, axis=1)
        return x


