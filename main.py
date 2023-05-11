import lightning_model
import lightning_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback


if __name__ == "__main__":
    
    
    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode='min')
    
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=False,
    mode='min'
    )
    
    yolo_model = lightning_model.YoloModule()
    
    data_model = lightning_data.MadaiModule()
    data_model.setup()
    

    trainer = pl.Trainer(devices=1, max_epochs=3)

    trainer.fit(model = yolo_model, datamodule=data_model)

    trainer.test(model = yolo_model, datamodule=data_model)