"""
Nazwa: lightning_model.py
Opis: Główny LightningModule modelu.
Autor: Jan Walczak, Bartłomiej Moroz
"""
from typing import Dict, List, Tuple
import pytorch_lightning as pl
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from yolo import YOLOv3
from yolo_post import YOLOProcessor
import wandb
import loading_weights
import nms


class YOLOv3Module(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, weight_decay=0, obj_coeff=1, noobj_coeff=100, ignore_threshold=0.5, weights_file='weights/darknet53.conv.74', load_only_backbone=True) -> None:
        super().__init__()
        # hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.obj_coeff = obj_coeff
        self.noobj_coeff = noobj_coeff
        self.ignore_threshold = ignore_threshold
        # Init model & helpers
        self.img_size = 416
        self.model = YOLOv3(self.num_classes)
        self.proc_52 = YOLOProcessor([(10, 13), (16, 30), (33, 23)], 8, self.img_size, self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)
        self.proc_26 = YOLOProcessor([(30, 61), (62, 45), (59, 119)], 16, self.img_size, self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)
        self.proc_13 = YOLOProcessor([(116, 90), (156, 198), (373, 326)], 32, self.img_size, self.num_classes, obj_coeff, noobj_coeff, ignore_threshold)
        # Loading model weights
        if load_only_backbone:
            loading_weights.load_model_parameters(weights_file, self.model.backbone)
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            loading_weights.load_model_parameters(weights_file, self.model)
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _common_step(self, batch: list, batch_idx: int, mode: str) -> Tuple[torch.Tensor, float]:
        """
        Run model forward pass and, depending on supplied mode, execute some post-processing functions.
        `train`:          forward, reshape_and_sigmoid, loss, process_after_loss, _
        `test` and `val`: forward, reshape_and_sigmoid, loss, process_after_loss, reduce_boxes
        `predict`:        forward, reshape_and_sigmoid, _,    process_after_loss, reduce_boxes
        """
        img_tensor, targets, _, _, = batch
        loss = 0
        outputs = list(self(img_tensor))
        for i, proc in enumerate((self.proc_13, self.proc_26, self.proc_52)):
            x = outputs[i]
            # Apply sigmoid function on x, y, objectness, classes and also reshape into (batches, anchors, grid_size, grid_size, outputs)
            x = proc.reshape_and_sigmoid(x)
            # Calculate YOLO loss function
            loss += proc.loss(x, targets) if mode != 'predict' else 0
            # Get real bounding boxes and reshape into (batches, all_boxes, outputs)
            x = proc.process_after_loss(x)
            outputs[i] = x
        preds = torch.cat(outputs, dim=1)
        # Even more post-processing
        results = nms.reduce_boxes(preds, min_max_size=(2, self.img_size)) if mode != 'train' else preds
        return results, loss

    def _common_test_valid_step(self, batch: list, batch_idx: int, mode: str) -> Tuple[float, dict]:
        """
        Run _common_step and calculate mAP
        TODO: Move mAP to _common_step and activate only for train and val?
        """
        _, targets, _, _, = batch
        results, loss = self._common_step(batch, batch_idx, mode)
        mAP_dict = self._calc_mAP(results, targets)
        return loss, mAP_dict

    def predict_step(self, batch: list, batch_idx: int):
        _, _, img_path, org_size, = batch
        results, _ = self._common_step(batch, batch_idx, mode='predict')
        return {'results': results, 'img_path': img_path, 'org_size': org_size}

    def training_step(self, batch, batch_idx: int) -> float:
        _, loss = self._common_step(batch, batch_idx, mode='train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)      
        return loss

    def test_step(self, batch, batch_idx: int) -> Dict[str, float]:
        loss, mAP_dict = self._common_test_valid_step(batch, batch_idx, mode='test')
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mAP', mAP_dict['map'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mAP50', mAP_dict['map_50'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_mAP': mAP_dict['map'].item(), 'test_mAP50': mAP_dict['map_50'].item()}

    def validation_step(self, batch, batch_idx: int) -> Dict[str, float]:
        loss, mAP_dict = self._common_test_valid_step(batch, batch_idx, mode='val')
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mAP', mAP_dict['map'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mAP50', mAP_dict['map_50'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mAP': mAP_dict['map'].item(), 'val_mAP50': mAP_dict['map_50'].item()}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calc_mAP(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate mAP for predictions. Returns a dictionary with various mAPs and mARs (unused).
        """
        map = MeanAveragePrecision()
        preds = []
        tgts = []
        grouped_predictions = self._group_by_batch_idx(predictions, predictions[..., 0])
        grouped_targets = self._group_by_batch_idx(targets, targets[..., 0])
        for img_preds, img_targets in zip(grouped_predictions, grouped_targets):
            preds.append({'boxes': img_preds[..., 1:5], 'scores': img_preds[..., 5], 'labels': img_preds[..., 6]})
            tgts.append({'boxes': img_targets[..., 2:6], 'labels': img_targets[..., 1]})
        map.update(preds, tgts)
        map_dict = map.compute()
        return map_dict

    def _group_by_batch_idx(self, tensor: torch.Tensor, indices: torch.Tensor) -> List[torch.Tensor]:
        """
        Very loosely based on https://github.com/pytorch/pytorch/issues/20613
        """
        _, order = indices.sort(0)
        delta = indices[1:] - indices[:-1]
        cutpoints = (delta.nonzero() + 1).tolist()
        cutpoints.insert(0, [0])
        cutpoints.append([len(indices)])
        res = []
        for i in range(len(cutpoints) - 1):
            res.append(tensor[order[cutpoints[i][0]:cutpoints[i+1][0]]])
        return res
