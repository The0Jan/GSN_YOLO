import torch
import torch.nn as nn
import numpy as np


class Convolutional(nn.Module):
    """
    Basic Darknet Convolutional unit containing a 2D Convolutional layer,
    a normalization layer and leaky ReLU activation layer.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Residual(nn.Module):
    """
    Residual layer containing 1x1 and 3x3 convolutional blocks and an additive
    skip connection.
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Convolutional(in_channels, in_channels // 2, kernel_size=1, padding=0),
            Convolutional(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual


class ResidualBlock(nn.Module):
    """
    Block of repeated Residual layers.
    """
    def __init__(self, repeat, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Convolutional(in_channels, out_channels, kernel_size=3, stride=2, padding = 0),
            nn.Sequential(*[Residual(out_channels) for _ in range(repeat)]),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Darknet53(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.route_con_1 
        self.route_con_2
        # Starting point
        self.conv = Convolutional(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1)
        # Residual blocks
        self.block_1 = ResidualBlock(repeat=1, in_channels=32,  out_channels=64)
        self.block_2 = ResidualBlock(repeat=2, in_channels=64,  out_channels=128)
        self.block_3 = ResidualBlock(repeat=8, in_channels=128, out_channels=256)
        self.block_4 = ResidualBlock(repeat=8, in_channels=256, out_channels=512)
        self.block_5 = ResidualBlock(repeat=4, in_channels=512, out_channels=1024)
        # Fully connected layer omitted


    def forward(self, x):
        # Starting point
        x = self.conv(x)
        # Residual blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        self.route_con_1 = x
        x = self.block_4(x)
        self.route_con_2 = x
        x = self.block_5(x)
        # Fully connected layer omitted
        return x
