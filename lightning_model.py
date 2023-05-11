import pytorch_lightning as pl
import torch
import yolo
import wandb
import loading_weights
import nms


class YOLOv3Module(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, weights_file='weights/darknet53.conv.74') -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = yolo.YOLOv3(self.num_classes)
        loading_weights.load_model_parameters(weights_file, self.model.darknet)
        for param in self.model.darknet.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds, _ = self.model(x, targets)
        results = nms.after_party(preds)
        return results

    def training_step(self, batch, batch_idx):
        img_tensor, target, img_path, org_size, = batch
        _, loss = self.model(img_tensor, target)
        self.log("train_loss", loss)        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
