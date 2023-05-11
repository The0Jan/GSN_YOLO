import darknet
import torch
import torch.nn as nn
from typing import List, Tuple
import torchvision

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
            for i in range(repeat - 1)]
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
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class ConvolutionalUpsample(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor=2, mode='nearest') -> None:
        super().__init__()
        self.block = nn.Sequential(
            darknet.Convolutional(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class YOLODetector(nn.Module):
    """
    """
    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, obj_coff=1, noobj_coff=100, ignore_threshold=0.5) -> None:
        super().__init__()
        self.num_outputs = num_classes + 4 + 1
        self.num_anchors = len(anchors)
        self.grid = None
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # We set stride in forward()
        self.stride = None
        # Turn array of (width, height) anchor sizes into a tensor
        # Achieve that by flattening anchor sizes into a 1D list, turning into a tensor,
        # then reshaping into (batches, anchors, width, height, values)
        self.anchors = torch.tensor([x for anchor in anchors for x in anchor]).float().view(1, -1, 1, 1, 2)
        self.are_anchors_scaled = False
        self.ignore_threshold = ignore_threshold
        self.obj_coff = obj_coff
        self.noobj_coff = noobj_coff

    def _make_grid(self, width: int, height: int, device: torch.device) -> torch.Tensor:
        """
        Create a 1x1x`height`x`width`x2 tensor containing coordinates of a grid.
        [[[[0,0], [0,1]],
          [[1,0], [1,1]]]] etc.
        """
        x, y = torch.meshgrid([torch.arange(width), torch.arange(height)], indexing='ij')
        return torch.stack((x, y), dim=2).view(1, 1, width, height, 2).float().to(device)

    def forward(self, x: torch.Tensor, targets: torch.Tensor, img_size: int) -> Tuple[torch.Tensor, float]:
        # Stride is pixels per cell
        self.stride = img_size // x.size(2)
        # Reshape output tensor
        # from (batches, all_outputs, width, height) into (batches, anchors, width, height, outputs)
        batches, _, width, height = x.shape
        x = x.view(batches, self.num_anchors, self.num_outputs, width, height).permute(0, 1, 3, 4, 2).contiguous()
        # Create a 2D grid of cells (once only)
        if self.grid is None or self.grid.shape[2:4] != x.shape[2:4]:
            self.grid = self._make_grid(width, height, x.device)
        # Scale anchors (once only)
        if not self.are_anchors_scaled:
            self.anchors /= self.stride
            self.anchors = self.anchors.to(x.device)
            self.are_anchors_scaled = True
        # Sigmoids
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 4:] = torch.sigmoid(x[..., 4:])
        # Calculate loss if applicable
        loss = self.loss(x, targets) if targets is not None else 0
        # Find final bounding boxes
        # tx, ty, tw, th = predicted tensor
        # cx, cy = grid offset from top-left (in cells)
        # aw, ah = constant anchor sizes (in pixels)
        # (x, y) = sigmoid((tx, ty)) + (cx, cy)
        # (w, h) = exp((tw, th)) * (aw, ah)
        # conf   = sigmoid(conf)
        # cls    = sigmoid(cls)
        predictions = x.clone().detach()
        predictions[..., 0:2] = x[..., 0:2] + self.grid               # anchor x, y (in cells)
        predictions[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchors # anchor width, height (in cells)
        predictions[..., 4:] = x[..., 4:]                             # confidence, class
        # Turn bbox (in cells) to bbox (in pixels)
        predictions[..., :4] *= self.stride
        # Final shape is just a batched list of outputs
        predictions = predictions.view(batches, -1, self.num_outputs)
        return predictions, loss

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        truths, obj_mask, noobj_mask, class_mask, ious = self.transform_truths(predictions, targets, self.anchors.view(-1, 2), self.ignore_threshold)
        # Bounding box loss
        loss_x = self.mse_loss(predictions[..., 0][obj_mask], truths[..., 0][obj_mask])
        loss_y = self.mse_loss(predictions[..., 1][obj_mask], truths[..., 1][obj_mask])
        loss_w = self.mse_loss(predictions[..., 2][obj_mask], truths[..., 2][obj_mask])
        loss_h = self.mse_loss(predictions[..., 3][obj_mask], truths[..., 3][obj_mask])
        # Objectness loss
        loss_obj = self.bce_loss(predictions[..., 4][obj_mask], truths[..., 4][obj_mask])
        loss_noobj = self.bce_loss(predictions[..., 4][noobj_mask], truths[..., 4][noobj_mask])
        # Classification loss
        loss_class = self.bce_loss(predictions[..., 5:][obj_mask], truths[..., 5:][obj_mask])
        return loss_x + loss_y + loss_w + loss_h\
             + self.obj_coff * loss_obj + self.noobj_coff * loss_noobj\
             + loss_class

    def transform_truths(self, predictions: torch.Tensor, targets: torch.Tensor,
                         anchors: torch.Tensor, ignore_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tensor of same shape as predictions (batches, anchors, grid, grid, outputs) but with different outputs.
        # Outputs are (x, y, w, h, conf, classes) like predictions, but only filled for elements that best match ground truth.
        # x, y are between 0 and 1
        # w, h are between log(0) and log(grid)
        # obj is 1 for best anchor match, 0 for others
        # class is one-hot encoded (1 for matching class, 0 for others)
        GRID_SIZE = predictions.size(2)
        EPS = targets.new_ones(targets.size(0), 2) * 1e-10
        truths = predictions.new_zeros(predictions.shape).float()
        # Mask of best anchor fits
        obj_mask = predictions.new_zeros(predictions.shape[:4]).bool()
        # Opposite of obj_mask that also has zeroes when IoU of truth and anchors is higher that ignore_threshold
        noobj_mask = predictions.new_ones(obj_mask.shape).bool()
        # Mask of best anchor fits that also have matching classes
        class_mask = predictions.new_zeros(obj_mask.shape).float()
        # Tensor of some IoUs for something ???
        ious = predictions.new_zeros(obj_mask.shape).float()
        # Unpack ground truths
        img_idx, t_class = targets[..., :2].long().t()
        # Box coordinates are scaled to (0, 416), but we want (0, GRID_SIZE)
        targets[..., 2:6] /= self.stride
        cxy, bwh = xyxy2xywh(targets[..., 2:6])
        # IoUs of anchors and true boxes
        # Anchors are in (w, h) format, we want (x1, y1, x2, y2), so let's move true boxes to 0, 0
        # and make anchors (0, 0, w, h) to calculate IoU.
        # We can do that because we don't care about positions, just which boxes have similar size.
        anchors2 = torch.cat([anchors.new_zeros(anchors.shape), anchors], dim=1)
        moved_targets = targets.clone()
        moved_targets[..., 2:4] *= 0
        moved_targets[..., 4:6] -= targets[..., 2:4]
        anchor_ious = torchvision.ops.box_iou(anchors2, moved_targets[..., 2:6])
        best_ious, best_anchors = torch.max(anchor_ious, dim=0)
        # i, j = floor(x, y), aka just grid coords
        gij = cxy.long()
        # Clamp values to stay within grid size
        gij[gij[..., 0] < 0] = 0
        gij[gij[..., 1] < 0] = 0
        gij[gij[..., 0] >= GRID_SIZE] = GRID_SIZE - 1
        gij[gij[..., 1] >= GRID_SIZE] = GRID_SIZE - 1
        # set obj and noobj masks
        obj_mask[img_idx, best_anchors, gij[..., 1], gij[..., 0]] = 1
        noobj_mask[img_idx, best_anchors, gij[..., 1], gij[..., 0]] = 0
        # Also set noobj_mask to 0 when iou is above threshold
        for i, iou in enumerate(anchor_ious.t()):
            noobj_mask[img_idx[i], iou > ignore_threshold, gij[i, 1], gij[i, 0]] = 0
        # Set output x, y
        truths[img_idx, best_anchors, gij[..., 1], gij[..., 0], 0:2] = cxy - cxy.floor()
        # Set output w, h
        truths[img_idx, best_anchors, gij[..., 1], gij[..., 0], 2:4] = torch.log(bwh / anchors[best_anchors] + EPS)
        # Anchor match in truths
        truths[..., 4] = obj_mask.float()
        # One-hot encoding class
        truths[img_idx, best_anchors, gij[..., 1], gij[..., 0], t_class + 5] = 1
        # Class correctness and IoU of matched anchors - not for loss, but for later metrics
        pred_xywh = predictions[img_idx, best_anchors, gij[..., 1], gij[..., 0], :4]
        pred_class_confs = predictions[img_idx, best_anchors, gij[..., 1], gij[..., 0], 5:]
        pred_conf, pred_class = torch.max(pred_class_confs, dim=-1)
        class_mask[img_idx, best_anchors, gij[..., 1], gij[..., 0]] = (pred_class == t_class).float()
        # box_iou gives cartesian product, but we only want comparisons with the same row, so we use diagonal
        ious[img_idx, best_anchors, gij[..., 1], gij[..., 0]] = torchvision.ops.box_iou(xywh2xyxy(pred_xywh), targets[..., 2:6]).diag()
        # Revert from (0, GRID_SIZE) to -> (0, 416) (otherwise targets end up modified)
        targets[..., 2:6] *= self.stride
        return truths, obj_mask, noobj_mask, class_mask, ious


def xyxy2xywh(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bwh = boxes.new(boxes.size(0), 2)
    cxy = boxes.new(bwh.shape)
    for i in range(2):
        bwh[..., i] = boxes[..., i+2] - boxes[..., i]
        cxy[..., i] = boxes[..., i] + bwh[..., i] / 2
    return cxy, bwh


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    new_boxes = boxes.new(boxes.shape)
    for i in range(2):
        new_boxes[..., i]   = boxes[..., i] - boxes[..., i+2] / 2
        new_boxes[..., i+2] = boxes[..., i] + boxes[..., i+2] / 2
    return new_boxes


class YOLOv3(nn.Module):
    def __init__(self, num_classes: int, in_channels=3, bounding_boxes=3, obj_coeff=1, noobj_coeff=100, ignore_threshold=0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.predictions_total = bounding_boxes * (4 + 1 + self.num_classes)
        ### Backbone
        self.backbone = darknet.Darknet53(in_channels)
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
        ### Head detector layers
        self.head_0 = YOLODetector([(116, 90), (156, 198), (373, 326)], self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)
        self.head_1 = YOLODetector([(30, 61), (62, 45), (59, 119)], self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)
        self.head_2 = YOLODetector([(10, 13), (16, 30), (33, 23)], self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        results = []
        total_loss = 0
        width = x.size(2)
        ### Backbone
        x52, x26, x13 = self.backbone(x)
        ### 13x13
        x = self.conv_block_0(x13)
        out, loss = self.head_0(self.conv_0_f(x), targets, width)
        total_loss += loss
        results.append(out)
        ### 26x26
        x = self.conv_and_upsample_1(x)
        x = torch.cat([x, x26], dim=1)
        x = self.conv_block_1(x)
        out, loss = self.head_1(self.conv_1_f(x), targets, width)
        total_loss += loss
        results.append(out)
        ### 52x52
        x = self.conv_and_upsample_2(x)
        x = torch.cat([x, x52], dim=1)
        x = self.conv_block_2(x)
        out, loss = self.head_2(self.conv_2_f(x), targets, width)
        total_loss += loss
        results.append(out)
        # Finally, results
        return torch.cat(results, dim=1), total_loss
