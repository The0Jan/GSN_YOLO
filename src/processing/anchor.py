"""
Nazwa: anchor.py
Opis: Przetwarzanie surowego wyjścia z YOLO: funkcja straty,
      dopasowywanie anchorów.
Autor: Bartłomiej Moroz
"""
import torch
import torch.nn as nn
import torchvision
from typing import List, Tuple


class YOLOProcessor:
    """
    Helper class that processes model outputs for one prediction head.
    """

    def __init__(
        self,
        anchors: List[Tuple[int, int]],
        stride: int,
        img_size: int,
        num_classes: int,
        obj_coff=1,
        noobj_coff=100,
        ignore_threshold=0.5,
    ) -> None:
        super().__init__()
        self.num_outputs = num_classes + 4 + 1
        self.num_anchors = len(anchors)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.stride = stride
        self.img_size = img_size
        # These three are used in loss()
        self.ignore_threshold = ignore_threshold
        self.obj_coff = obj_coff
        self.noobj_coff = noobj_coff
        # Create a 2D grid of cells
        self.grid = self._make_grid(self.img_size // self.stride)
        # Turn array of (width, height) anchor sizes (in pixels) into a tensor with (width, height) in cells.
        # Achieve that by flattening anchor sizes into a 1D list, turning into a tensor,
        # then reshaping into (batches, anchors, grid_size, grid_size, values)
        self.anchors = (
            torch.tensor([x / self.stride for anchor in anchors for x in anchor])
            .float()
            .view(1, -1, 1, 1, 2)
        )

    def reshape_and_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape output tensor and apply sigmoid function on x, y coordinates, objectness and classes.
        """
        self.anchors = self.anchors.to(x.device)
        # Reshape output tensor
        # from (batches, all_outputs, grid_size, grid_size) into (batches, anchors, grid_size, grid_size, outputs)
        batches, _, grid_size, grid_size = x.shape
        x = (
            x.view(batches, self.num_anchors, self.num_outputs, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # Sigmoids
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])
        x[..., 4:] = torch.sigmoid(x[..., 4:])
        return x

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate loss for given batch and targets. Input targets are in (class, x1, y1, x2, y2)
        format, where coordinates are in [0, 416) range.
        """
        # Transform ground truths from "human" format to one matching raw predictions output.
        truths, obj_mask, noobj_mask = self._transform_truths(
            predictions, targets, self.anchors.view(-1, 2), self.ignore_threshold
        )
        # Bounding box loss
        loss_x = self.mse_loss(predictions[..., 0][obj_mask], truths[..., 0][obj_mask])
        loss_y = self.mse_loss(predictions[..., 1][obj_mask], truths[..., 1][obj_mask])
        loss_w = self.mse_loss(predictions[..., 2][obj_mask], truths[..., 2][obj_mask])
        loss_h = self.mse_loss(predictions[..., 3][obj_mask], truths[..., 3][obj_mask])
        # Objectness loss
        loss_obj = self.bce_loss(
            predictions[..., 4][obj_mask], truths[..., 4][obj_mask]
        )
        loss_noobj = self.bce_loss(
            predictions[..., 4][noobj_mask], truths[..., 4][noobj_mask]
        )
        # Classification loss
        loss_class = self.bce_loss(
            predictions[..., 5:][obj_mask], truths[..., 5:][obj_mask]
        )
        return (
            loss_x
            + loss_y
            + loss_w
            + loss_h
            + self.obj_coff * loss_obj
            + self.noobj_coff * loss_noobj
            + loss_class
        )

    def process_after_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find final bounding boxes by adjusting anchors by model output and flatten output
        into (batch_size, -1, outputs).
        """
        self.grid = self.grid.to(x.device)
        # tx, ty, tw, th = predicted tensor
        # cx, cy = grid offset from top-left (in cells)
        # aw, ah = constant anchor sizes (in cells)
        # (x, y) = (tx, ty) + (cx, cy)
        # (w, h) = exp((tw, th)) * (aw, ah)
        # Clone and detch, because we don't want this in our gradient propagation.
        predictions = x.clone().detach()
        predictions[..., 0:2] = x[..., 0:2] + self.grid  # anchor x, y (in cells)
        predictions[..., 2:4] = (
            torch.exp(x[..., 2:4]) * self.anchors
        )  # anchor width, height (in cells)
        predictions[..., 4:] = x[..., 4:]  # confidence, class
        # Turn bbox (in cells) to bbox (in pixels)
        predictions[..., :4] *= self.stride
        # Final shape is just a batched list of outputs
        predictions = predictions.view(x.size(0), -1, self.num_outputs)
        return predictions

    def _transform_truths(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        anchors: torch.Tensor,
        ignore_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform ground truths from "human readable" format of (class, x1, y1, x2, y2) to format
        that matches raw model predictions, such that loss can be calculated easily.
        Based on: https://github.com/v-iashin/WebsiteYOLO/blob/master/darknet.py#L237 (REALLY nice comments)
        """
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
        # Unpack ground truths
        img_idx, t_class = targets[..., :2].long().t()
        # Box coordinates are scaled to (0, 416), but we want (0, GRID_SIZE)
        targets[..., 2:6] /= self.stride
        cxy, bwh = self._xyxy2xywh(targets[..., 2:6])
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
        truths[img_idx, best_anchors, gij[..., 1], gij[..., 0], 2:4] = torch.log(
            bwh / anchors[best_anchors] + EPS
        )
        # Anchor match in truths
        truths[..., 4] = obj_mask.float()
        # One-hot encoding class
        truths[img_idx, best_anchors, gij[..., 1], gij[..., 0], t_class + 5] = 1
        # Revert from (0, GRID_SIZE) to -> (0, 416) (otherwise targets end up modified)
        targets[..., 2:6] *= self.stride
        return truths, obj_mask, noobj_mask

    def _make_grid(self, grid_size: int) -> torch.Tensor:
        """
        Create a 1x1x`grid_size`x`grid_size`x2 tensor containing coordinates of a grid.
        [[[[0,0], [1,0]],
          [[0,1], [1,1]]]] etc.
        """
        x, y = torch.meshgrid(
            [torch.arange(grid_size), torch.arange(grid_size)], indexing="ij"
        )
        return (
            torch.stack((x, y), dim=2)
            .view(1, 1, grid_size, grid_size, 2)
            .float()[..., [1, 0]]
        )

    def _xyxy2xywh(self, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bwh = boxes.new(boxes.size(0), 2)
        cxy = boxes.new(bwh.shape)
        for i in range(2):
            bwh[..., i] = boxes[..., i + 2] - boxes[..., i]
            cxy[..., i] = boxes[..., i] + bwh[..., i] / 2
        return cxy, bwh

    def _xywh2xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        new_boxes = boxes.new(boxes.shape)
        for i in range(2):
            new_boxes[..., i] = boxes[..., i] - boxes[..., i + 2] / 2
            new_boxes[..., i + 2] = boxes[..., i] + boxes[..., i + 2] / 2
        return new_boxes
