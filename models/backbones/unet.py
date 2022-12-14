import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_x2(x)


class DownScale(nn.Module):
    """
    Downscaling with maxpool then double-conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpScale(nn.Module):
    """
    Upscaling then double-conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        #   https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        #   https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.conv_in = DoubleConv(n_channels, 64)
        self.down1 = DownScale( 64, 128)
        self.down2 = DownScale(128, 256)
        self.down3 = DownScale(256, 512)
        self.down4 = DownScale(512, 1024 // factor)

        self.up1 = UpScale(1024, 512 // factor, bilinear)
        self.up2 = UpScale( 512, 256 // factor, bilinear)
        self.up3 = UpScale( 256, 128 // factor, bilinear)
        self.up4 = UpScale( 128,  64          , bilinear)

    def forward(self, x):
        Ys = [x]
        Ys.append(self.conv_in(Ys[-1]))
        Ys.append(self.down1(Ys[-1]))
        Ys.append(self.down2(Ys[-1]))
        Ys.append(self.down3(Ys[-1]))
        Ys.append(self.down4(Ys[-1]))
        Ys.append(self.up1(Ys[-1], Ys[4]))
        Ys.append(self.up2(Ys[-1], Ys[3]))
        Ys.append(self.up3(Ys[-1], Ys[2]))
        Ys.append(self.up4(Ys[-1], Ys[1]))
        return Ys













