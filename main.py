from lightning_model import YOLOv3Module 
from lightning_data import MADAIDataModule
from primitive_dataloader import PrimitiveDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
import dataset
import argparse
import os
import gdown


def predict(model, datamodule):
    for i, batch in enumerate(datamodule.predict_dataloader()):
        y = model.predict_step(batch, i)
        r = y['results']
        idx = 0
        dataset.visualize_results(y['img_path'][idx], r[r[..., 0] == idx, :].tolist())
        break


def parse_args():
    parser = argparse.ArgumentParser(description='YoloV3')
    parser.add_argument('mode', type=str, choices=['train', 'test', 'predict'], help='Mode of action')
    parser.add_argument('-l', '--logger', type=str, choices=['csv', 'tensorboard', 'wandb'], default='csv', help='Logger to use')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Load model checkpoint from file')
    parser.add_argument('-e','--earlystop', type=bool, default=True, help='Should use early stop callback?')
    return parser.parse_args()


def download(weights_dir: str, model_dir: str):
    # Download pretrained weights
    weights_gid = "1mL2BC_UM3iHXMZFmKgrlO9jP5yVdYMbr"
    weights_file = "darknet53.conv.74"
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    weights_file = os.path.join(weights_dir, weights_file)
    if not os.path.isfile(weights_file):
        gdown.download(id=weights_gid, output=weights_file)
    # Download best model checkpoint
    model_gid = "16x4pcp_mWr-MSv0TcliAGyfk843EjXWp"
    model_dir = "model"
    model_file = "best.ckpt"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_file = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_file):
        gdown.download(id=model_gid, output=model_file)


def main(args):
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
    # Download files
    download(weights_dir="weights", model_dir=MODEL_CKPT_PATH)
    # Choose logger
    logger = None
    if args.logger == 'csv':
       logger = CSVLogger('csv_logs')
    elif args.logger == 'tensorboard':
        logger = TensorBoardLogger(save_dir='tensorboard')
    elif args.logger == 'wandb':
        logger = WandbLogger(project='gsn-YOLOv3', job_type='train')
    # Choose model or model checkpoint
    if args.checkpoint is not None:
        yolo_model = YOLOv3Module.load_from_checkpoint(MODEL_CKPT_PATH + args.checkpoint)
    else:
        yolo_model = YOLOv3Module(num_classes=5)
    # Set up callbacks
    callbacks = [checkpoint_callback]
    if args.earlystop == True:
        callbacks.append(early_stop_callback)
    # Init DataModule and Trainer
    if args.mode != 'predict':
        data_model = MADAIDataModule(batch_size=8, num_workers=0) # NOTE: num_workers _must_ be 0 for now, else it causes an instant error
    else:
        data_model = PrimitiveDataModule(None, 'test-new/images', batch_size=8, num_workers=0)
        data_model.setup()
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=30, callbacks=callbacks, logger=logger)
    # Begin work
    if args.mode == 'train':
        trainer.fit(model=yolo_model, datamodule=data_model)   
    elif args.mode == 'test':
        trainer.test(model=yolo_model, datamodule=data_model)
    elif args.mode == 'predict':
        predict(model=yolo_model, datamodule=data_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
