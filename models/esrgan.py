import torch
from torch import nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels:int, increased_channels:int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.cnv1 = nn.Conv2d(in_channel + increased_channels*1, in_channels)
        self.cnv2 = nn.Conv2d(in_channels, increased_channels)
        self.cnv1 = nn.Conv2d(in_channels, increased_channels)
        self.cnv1 = nn.Conv2d(in_channels, increased_channels)
        self.cnv1 = nn.Conv2d(in_channels, increased_channels)