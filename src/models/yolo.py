"""
Nazwa: yolo.py
Opis: Model YOLOv3.
Autor: Bartłomiej Moroz
"""
from src.models.darknet import Convolutional, Darknet53
import torch
import torch.nn as nn
from typing import Tuple

class ConvolutionalBlock(nn.Module):
    """
    A block of repeated Darknet Convolutional layers with alternating input/output sizes and kernel sizes.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, repeat=5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Convolutional(in_channels, out_channels, kernel_size=1),
            *[
                Convolutional(out_channels, mid_channels, kernel_size=3) if i % 2 == 0 else
                Convolutional(mid_channels, out_channels, kernel_size=1)
            for i in range(repeat - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class FinalConvolutional(nn.Module):
    """
    Final prediction head. Turns image features into model outputs.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Convolutional(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class ConvolutionalUpsample(nn.Module):
    """
    Upsampling layer.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor=2, mode='nearest') -> None:
        super().__init__()
        self.block = nn.Sequential(
            Convolutional(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class YOLOv3(nn.Module):
    """
    Complete YOLOv3 module. See yolo.drawio diagram for details.
    """
    def __init__(self, num_classes: int, in_channels=3, bounding_boxes=3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.predictions_total = bounding_boxes * (4 + 1 + self.num_classes)
        ### Backbone
        self.backbone = Darknet53(in_channels)
        ### Features for 13x13 grid - for detecing large objects
        self.conv_block_0 = ConvolutionalBlock(in_channels=1024, mid_channels=1024, out_channels=512, repeat=5)
        self.conv_0_f = FinalConvolutional(in_channels=512, mid_channels=1024, out_channels=self.predictions_total)
        ### Features for 26x26 grid - for detecting medium objects
        self.conv_and_upsample_1 = ConvolutionalUpsample(512, 256)
        self.conv_block_1 = ConvolutionalBlock(in_channels=512+256, mid_channels=512, out_channels=256, repeat=5)
        self.conv_1_f = FinalConvolutional(in_channels=256, mid_channels=512, out_channels=self.predictions_total)
        ### Features for 52x52 grid - for detecting small objects
        self.conv_and_upsample_2 = ConvolutionalUpsample(256, 128)
        self.conv_block_2 = ConvolutionalBlock(in_channels=256+128, mid_channels=256, out_channels=128, repeat=5)
        self.conv_2_f = FinalConvolutional(in_channels=128, mid_channels=256, out_channels=self.predictions_total)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        results = []
        ### Backbone
        x52, x26, x13 = self.backbone(x)
        ### 13x13
        x = self.conv_block_0(x13)
        out = self.conv_0_f(x)
        results.append(out)
        ### 26x26
        x = self.conv_and_upsample_1(x)
        x = torch.cat([x, x26], dim=1)
        x = self.conv_block_1(x)
        out = self.conv_1_f(x)
        results.append(out)
        ### 52x52
        x = self.conv_and_upsample_2(x)
        x = torch.cat([x, x52], dim=1)
        x = self.conv_block_2(x)
        out = self.conv_2_f(x)
        results.append(out)
        # Finally, results
        return tuple(results)
