import dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import yolo

class YoloModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.yolo = yolo.YOLOv3(5)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        results = self.yolo(x)
        return results

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, org_size, y = batch
        z = self.yolo(x)
        ### Tutaj magia naszych funkcji do obliczania błędu
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        
        ###
        
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer