"""
Nazwa: darknet.py
Opis: Model Darknet53.
Autor: Jan Walczak (alpha), Bartłomiej Moroz (final)
"""
import torch
import torch.nn as nn
from typing import Tuple


class Convolutional(nn.Module):
    """
    Basic Darknet Convolutional unit containing a 2D Convolutional layer,
    a normalization layer and leaky ReLU activation layer.
    Kernel size of 3 implies padding of 1, else we get mismatched output shapes.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=1, **kwargs
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                bias=False,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            nn.BatchNorm2d(out_channels, momentum=0.03),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            Convolutional(in_channels, in_channels // 2, kernel_size=1),
            Convolutional(in_channels // 2, in_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)
        return x + residual


class ResidualBlock(nn.Module):
    """
    Block of repeated Residual layers, prefixed with a x2 downsampling convolutional layer.
    """

    def __init__(self, repeat, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Convolutional(in_channels, out_channels, kernel_size=3, stride=2),
            nn.Sequential(*[Residual(out_channels) for _ in range(repeat)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class Darknet53(nn.Module):
    """
    Complete Darknet53 module. See darknet.drawio diagram for details.
    """

    def __init__(self, in_channels) -> None:
        super().__init__()
        # Starting point
        self.conv = Convolutional(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1
        )
        # Residual blocks
        self.block_1 = ResidualBlock(repeat=1, in_channels=32, out_channels=64)
        self.block_2 = ResidualBlock(repeat=2, in_channels=64, out_channels=128)
        self.block_3 = ResidualBlock(repeat=8, in_channels=128, out_channels=256)
        self.block_4 = ResidualBlock(repeat=8, in_channels=256, out_channels=512)
        self.block_5 = ResidualBlock(repeat=4, in_channels=512, out_channels=1024)
        # Fully connected layer omitted

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Starting point
        x = self.conv(x)
        # Residual blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x52 = x
        x = self.block_4(x)
        x26 = x
        x = self.block_5(x)
        x13 = x
        # Fully connected layer omitted
        return x52, x26, x13
