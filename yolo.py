import darknet
import torch
import torch.nn as nn
from typing import List, Tuple

class ConvolutionalBlock(nn.Module):
    """
    A block of repeated Darknet Convolutional layers with alternating input/output sizes and kernel sizes.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, repeat=5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, out_channels, kernel_size=1),
            *[
                darknet.Convolutional(out_channels, mid_channels, kernel_size=3) if i % 2 == 0 else
                darknet.Convolutional(mid_channels, out_channels, kernel_size=1)
            for i in range(repeat-1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class FinalConvolutional(nn.Module):
    """
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class ConvolutionalUpsample(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, scale_factor=2, mode='nearest') -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, out_channels, kernel_size=kernel_size),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class YOLOLayer(nn.Module):
    """
    """
    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int) -> None:
        super().__init__()
        self.num_outputs = num_classes + 4 + 1
        self.num_anchors = len(anchors)
        # We set stride in forward()
        self.stride = None
        # Turn array of (width, height) anchor sizes into a tensor of same shape
        # Achieve that by flattening anchor sizes into a 1D list, turning into a tensor, then converting back into pairs
        anchors = torch.tensor([x for anchor in anchors for x in anchor]).float().view(-1, 2)
        # Anchor sizes are NOT a learning parameter, but we want them to persist, hence register_buffer
        # TODO check if this can be removed safely
        self.register_buffer('anchors', anchors)
        # Copy of anchor sizes resized into 1 x anchors x 1 x 1 x Point2D, which matches the inference output shape
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        # Stride is pixels per cell
        self.stride = img_size // x.size(2)
        # Reshape output tensor
        #   from: (batches, all_outputs, width, height)
        #   into: (batches, anchors, width, height, outputs)
        batches, _, width, height = x.shape
        x = x.view(batches, self.num_anchors, self.num_outputs, width, height).permute(0, 1, 3, 4, 2).contiguous()
        # In case of inference, present results directly
        if not self.training:
            # Create a 2D grid of cells (once only)
            if self.grid.shape[2:4] != x.shape[2:4]:
                vx, vy = torch.meshgrid([torch.arange(width), torch.arange(height)], indexing='ij')
                self.grid = torch.stack((vx, vy), dim=2).view(1, 1, width, height, 2).float().to(x.device)
            # Find final bounding box
            # tx, ty, tw, th = predicted tensor
            # cx, cy = grid offset from top-left (in cells)
            # pw, ph = constant anchor sizes (in pixels)
            # (x, y) = sigmoid((tx, ty)) + (cx, cy)
            # (w, h) = (pw, ph) * exp((tw, th))
            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * self.stride # anchor x, y (in px, not cells)
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid         # anchor width, height (in px)
            x[..., 4:] = x[..., 4:].sigmoid()                               # confidence, class
            # Final shape is just a list of outputs
            x = x.view(batches, -1, self.num_outputs)
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
        self.yolo_0 = YOLOLayer([(116, 90), (156, 198), (373, 326)], self.num_classes)
        ### Features for 26x26 grid - for detecting medium objects
        self.conv_and_upsample_1 = ConvolutionalUpsample(512, 256)
        self.conv_block_1 = ConvolutionalBlock(in_channels=512+256, mid_channels=512, out_channels=256, repeat=5)
        self.conv_1_f = FinalConvolutional(in_channels=256, mid_channels=512, out_channels=self.predictions_total)
        self.yolo_1 = YOLOLayer([(30, 61), (62, 45), (59, 119)], self.num_classes)
        ### Features for 52x52 grid - for detecting small objects
        self.conv_and_upsample_2 = ConvolutionalUpsample(256, 128)
        self.conv_block_2 = ConvolutionalBlock(in_channels=256+128, mid_channels=256, out_channels=128, repeat=5)
        self.conv_2_f = FinalConvolutional(in_channels=128, mid_channels=256, out_channels=self.predictions_total)
        self.yolo_2 = YOLOLayer([(10, 13), (16, 30), (33, 23)], self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        ### Backbone
        x = self.darknet(x)
        ### 13x13
        x = self.conv_block_0(x)
        out = self.yolo_0(self.conv_0_f(x))
        results.append(out)
        ### 26x26
        x = self.conv_and_upsample_1(x)
        x = torch.cat([x, self.darknet.post_block_4], dim=1)
        x = self.conv_block_1(x)
        out = self.yolo_1(self.conv_1_f(x))
        results.append(out)
        ### 52x52
        x = self.conv_and_upsample_2(x)
        x = torch.cat([x, self.darknet.post_block_3], dim=1)
        x = self.conv_block_2(x)
        out = self.yolo_2(self.conv_2_f(x))
        results.append(out)
        # Finally, results
        return results if self.training else torch.cat(results, dim=1)
