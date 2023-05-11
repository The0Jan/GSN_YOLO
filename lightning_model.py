import pytorch_lightning as pl
import torch
import yolo
import wandb
import loading_weights  as l_w
import nms
class YoloModule(pl.LightningModule):
    def __init__(self, num_classes, learning_rate = 1e-3, weights_file = 'weights/darknet53.conv.74'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.model = yolo.YOLOv3(self.num_classes)
        l_w.load_model_parameters(weights_file, self.model.darknet)
        for param in self.model.darknet.parameters():
            param.requires_grad = False
        

    def forward(self, x, targets):
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