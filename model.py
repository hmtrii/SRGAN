import torch
from torch import nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return torch.add(x, self.res_block(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.PixelShuffle(up_scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.up_block(x)


class Generator(nn.Module):
    def __init__(self, num_res=16):
        super(Generator, self).__init__()
        self.pre_res_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding='same'),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64, 64) for i in range(num_res)])
        self.pre_upsample_blocks = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(64, 2) for i in range(2)])
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding='same')

    def forward(self, x):
        in_res_blocks = self.pre_res_blocks(x)
        out_res_blocks = self.res_blocks(in_res_blocks)
        in_upsample_blocks = torch.add(out_res_blocks, self.pre_upsample_blocks(out_res_blocks))
        out_upsample_blocks = self.upsample_blocks(in_upsample_blocks)
        out = self.last_conv(out_upsample_blocks)
        return out


class Discriminator(nn.Module):
    def __init__(self, width, height) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        width_fm = math.ceil(math.ceil(math.ceil(math.ceil(width / 2)/2)/2)/2)
        height_fm = math.ceil(math.ceil(math.ceil(math.ceil(height / 2)/2)/2)/2)
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=(width_fm*height_fm*512), out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1),
        )

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        out = self.dense_layers(features)
        return out
