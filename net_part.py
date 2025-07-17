import imp
import torch.nn as nn
from unet.util import *
import torch.nn.functional as F


# ------------------------- #
#       ResBlock
# ------------------------- #

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
        dims=2,
        src_channels=None,
        use_conv=True,
        ):
        super().__init__()

        self.use_conv = use_conv
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.src_layers = nn.Sequential(
            nn.Linear(src_channels, self.out_channels),
            nn.LeakyReLU(),
        )

        self.out_layers = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x_src_emb):
        x = x_src_emb['h']
        src_emb = x_src_emb['src_i']

        h = self.in_layers(x)
        src_emb = self.src_layers(src_emb)

        while len(src_emb.shape) < len(h.shape):
            src_emb = src_emb[..., None]

        h = h + src_emb
        h = self.out_layers(h)
        out = self.skip_connection(x) + h

        return out


# -------------------------- #
#          down
# -------------------------- #
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x_src_emd):
        x = x_src_emd['h']
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, shape=None, bilinear=True):
        super().__init__()

        self.shape = shape
        if bilinear:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            
        else:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            )
            self.conv = DoubleConv(in_channels, out_channels)
            

    def forward(self, x1):
        x1 = self.up(x1)
        diffY = self.shape - x1.size()[2]
        diffX = self.shape - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return self.conv(x1)

