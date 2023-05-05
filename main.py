import lightning_model
import lightning_data
import pytorch_lightning as pl


if __name__ == "__main__":
    
    yolo_model = lightning_model.YoloModule()
    
    data_model = lightning_data.MadaiModule()
    data_model.setup()
    
    trainer = pl.Trainer(devices=1, max_epochs=3)

    trainer.fit(model = yolo_model, datamodule=data_model)

    trainer.test(model = yolo_model, datamodule=data_model)