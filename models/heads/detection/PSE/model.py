"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py
"""

from torch import nn


class PSEHead(nn.Module):
    """
    Shape-robust Text Detection with Progressive Scale Expansion (PSE) network
    """
    def __init__(self, in_channels, hidden_dim=256, out_channels=7, **kwargs):
        super(PSEHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        return {'maps': out}


        