from lightning_model import YOLOv3Module
from dataset import YOLODataset
import dataset
from yolo import YOLOv3
import yolo_post
from primitive_dataloader import PrimitiveDataModule
from lightning_data import MADAIDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import dataset
from loading_weights import load_model_parameters

if __name__ == "__main__":
    """
    yolo_model = YOLOv3Module(num_classes=80, weights_file='weights/yolov3.weights', load_only_backbone=False)
    data_model = MADAIDataModule(batch_size=8, num_workers=4)
    data_model.setup()
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=3, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=yolo_model, datamodule=data_model)   
    trainer.test(model=yolo_model, datamodule=data_model)
    """

    yolo_model = YOLOv3Module(num_classes=5)
    #data_model = PrimitiveDataModule(None, 'test-val2017', batch_size=1, num_workers=4)
    data_model = MADAIDataModule(batch_size=8, num_workers=4)
    data_model.setup()
    #trainer = pl.Trainer(accelerator='gpu', devices=1, limit_predict_batches=1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=3)
    #res_dict = trainer.predict(model=yolo_model, datamodule=data_model)[0]
    trainer.fit(model=yolo_model, datamodule=data_model)
    trainer.test(model=yolo_model, datamodule=data_model)
    #print(res_dict)
    #r = res_dict['results']
    #dataset.visualize_results(res_dict['img_path'][0], r[r[..., 0] == 0, :].tolist())

    #for i, batch in enumerate(data_model.predict_dataloader()):
    #    if i == 0:
    #        x, _, path, _ = batch
    #        x, path = x[0], path[0]
    #        x = x.unsqueeze(0)
    #        y = yolo_model.predict_step(batch, 0)
    #        print(y)
    #        r = y['results']
    #        dataset.visualize_results(y['img_path'][0], r[r[..., 0] == 0, :].tolist())
