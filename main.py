from lightning_model import YOLOv3Module 
from lightning_data import MADAIDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import torch


def main():
    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{train_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    yolo_model = YOLOv3Module(num_classes=5)
    data_model = MADAIDataModule(batch_size=8)
    data_model.setup()

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=3, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=yolo_model, datamodule=data_model)   
    trainer.test(model=yolo_model, datamodule=data_model)


if __name__ == "__main__":
    main()
