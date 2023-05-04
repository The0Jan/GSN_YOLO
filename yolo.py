import darknet
import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    """
    A block of repeated Darknet Convolutional layers with alternating input/output sizes and kernel sizes.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, repeat=5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, out_channels, kernel_size=1),
            *[
                darknet.Convolutional(out_channels, mid_channels, kernel_size=3 , padding =1) if i % 2 == 0 else
                darknet.Convolutional(mid_channels, out_channels, kernel_size=1)
            for i in range(repeat-1)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class FinalConvolutional(nn.Module):
    """
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, mid_channels, kernel_size=3, padding = 1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvolutionalUpsample(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, scale_factor=2, mode='nearest') -> None:
        super().__init__()
        self.conv = darknet.Convolutional(in_channels, out_channels, kernel_size=kernel_size)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class YOLOv3(nn.Module):
    def __init__(self, num_classes: int, in_channels=3, bounding_boxes=3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.predictions_total = bounding_boxes * (4 + 1 + self.num_classes)
        ### Backbone
        self.darknet = darknet.Darknet53(in_channels)
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

    def forward(self, x):
        results = []
        x = self.darknet(x)
        saved_1 = self.darknet.post_block_3
        saved_2 = self.darknet.post_block_4
        ### 1 Tensor
        x = self.conv_block_0(x)
        results.append(self.conv_0_f(x))
        ### 2 Tensor
        print("Tensor 2")
        x = self.conv_and_upsample_1(x)
        print(saved_2.size())
        print(x.size())

        x = torch.cat([x, saved_2], dim=1)
        x = self.conv_block_1(x)
        results.append(self.conv_1_f(x))
        ### 3 Tensor
        x = self.conv_and_upsample_2(x)
        x = torch.cat([x, saved_1], dim=1)
        x = self.conv_block_2(x)
        results.append(self.conv_2_f(x))
        return results
