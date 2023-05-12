from typing import Dict, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yolo
import wandb
import loading_weights
import nms


class YOLOv3Module(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, weight_decay=0, weights_file='weights/darknet53.conv.74') -> None:
        super().__init__()
        # hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        #
        self.model = yolo.YOLOv3(self.num_classes)
        loading_weights.load_model_parameters(weights_file, self.model.darknet)
        for param in self.model.darknet.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds, _ = self.model(x, targets)
        results = nms.reduce_boxes(preds)
        return results

    def training_step(self, batch, batch_idx) -> float:
        img_tensor, target, img_path, org_size, = batch
        _, loss = self.model(img_tensor, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      
        return loss

    def common_test_valid_step(self, batch, batch_idx) -> Tuple[float, float]:
        img_tensor, target, _, _ = batch
        pred, loss = self.model(img_tensor, target)
        pred = nms.reduce_boxes(pred)
        mAP = self.calc_mAP(pred, target)
        return loss, mAP

    def calc_mAP(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        map = MeanAveragePrecision()
        preds = []
        tgts = []
        grouped_predictions = group_by_batch_idx(predictions, predictions[..., 0])
        grouped_targets = group_by_batch_idx(targets, targets[..., 0])
        for img_preds, img_targets in zip(grouped_predictions, grouped_targets):
            preds.append({'boxes': img_preds[..., 1:5], 'scores': img_preds[..., 5], 'labels': img_preds[..., 6]})
            tgts.append({'boxes': img_targets[..., 2:6] * 416, 'labels': img_targets[..., 1]})
        map.update(preds, tgts)
        map_dict = map.compute()
        print(map_dict)
        return map_dict['map'].item()

    def test_step(self, batch, batch_idx) -> Dict[str, float]:
        loss, mAP = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mAP', mAP, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_mAP': mAP}

    def validation_step(self, batch, batch_idx) -> Dict[str, float]:
        loss, mAP = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mAP', mAP, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mAP': mAP}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def group_by_batch_idx(tensor: torch.Tensor, indices: torch.Tensor) -> List[torch.Tensor]:
    # Very loosely based on https://github.com/pytorch/pytorch/issues/20613
    _, order = indices.sort(0)
    delta = indices[1:] - indices[:-1]
    cutpoints = (delta.nonzero() + 1).tolist()
    cutpoints.insert(0, [0])
    cutpoints.append([len(indices)])
    res = []
    for i in range(len(cutpoints) - 1):
        res.append(tensor[order[cutpoints[i][0]:cutpoints[i+1][0]]])
    return res
