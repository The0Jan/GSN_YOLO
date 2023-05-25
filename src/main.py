"""
Nazwa: main.py
Opis: Pobieranie potrzebnych plików, uruchamianie modelu.
Autor: Bartłomiej Moroz, Jan Walczak
"""
import gdown
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from src.datamodules.madai import MADAIDataModule
from src.datamodules.primitive import PrimitiveDataModule
from src.lightning.yolo import YOLOv3Module
from src.other.visualizing_results import visualize_results


CHECKPOINT_GID_BEST = "16x4pcp_mWr-MSv0TcliAGyfk843EjXWp"
CHECKPOINT_GID_SECOND = "1IlvbUfkeNNjITpWeG3IsJPJSrg2nPEhf"


def predict(model, datamodule, output, batch_count):
    # Prepare output directory
    os.makedirs(output, exist_ok=True)
    batches = 0
    # Predict in batches in a simple way
    for batch_i, batch in enumerate(datamodule.predict_dataloader(shuffle=True)):
        if batches == batch_count:
            break
        y = model.predict_step(batch, batch_i)
        batches += 1
        # Visualize results and save to files
        for i in range(len(y["img_path"])):
            r = y["results"]
            visualize_results(y["img_path"][i], output, r[r[..., 0] == i, :].tolist())


def download(weights_dir: str, model_dir: str, model_gid: str):
    # Download pretrained weights
    weights_gid = "1mL2BC_UM3iHXMZFmKgrlO9jP5yVdYMbr"
    weights_file = "darknet53.conv.74"
    os.makedirs(weights_dir, exist_ok=True)
    weights_file = os.path.join(weights_dir, weights_file)
    if not os.path.isfile(weights_file):
        gdown.download(id=weights_gid, output=weights_file)
    # Download model checkpoint
    if model_gid is not None:
        model_dir = "model"
        model_file = model_gid + ".ckpt"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_file):
            gdown.download(id=model_gid, output=model_file)


def main(args):
    MODEL_CKPT_PATH = "model/"
    MODEL_CKPT = "model-{epoch:02d}-{train_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=3,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="train_loss", patience=3, verbose=False, mode="min"
    )
    # Download files
    download(
        weights_dir="weights", model_dir=MODEL_CKPT_PATH, model_gid=args.checkpoint_gid
    )
    # Choose logger
    logger = None
    if args.logger == "csv":
        logger = CSVLogger("csv_logs")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir="tensorboard")
    elif args.logger == "wandb":
        logger = WandbLogger(project="gsn-YOLOv3", job_type="train")
    # Choose model or model checkpoint
    if args.checkpoint is not None:
        yolo_model = YOLOv3Module.load_from_checkpoint(
            MODEL_CKPT_PATH + args.checkpoint
        )
    elif args.checkpoint_gid is not None:
        yolo_model = YOLOv3Module.load_from_checkpoint(
            MODEL_CKPT_PATH + args.checkpoint_gid + ".ckpt"
        )
    else:
        yolo_model = YOLOv3Module(num_classes=5)
    # Set up callbacks
    callbacks = [checkpoint_callback]
    if args.earlystop == True:
        callbacks.append(early_stop_callback)
    # Init DataModule and Trainer
    if args.mode != "predict":
        data_model = MADAIDataModule(batch_size=args.batch_size, num_workers=0)
    else:
        data_model = PrimitiveDataModule(
            None, args.input, batch_size=args.batch_size, num_workers=0
        )
        data_model.setup()
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=30, callbacks=callbacks, logger=logger
    )
    # Begin work
    if args.mode == "train":
        trainer.fit(model=yolo_model, datamodule=data_model)
    elif args.mode == "test":
        trainer.test(model=yolo_model, datamodule=data_model)
    elif args.mode == "predict":
        yolo_model.eval()
        predict(
            model=yolo_model,
            datamodule=data_model,
            output=args.output,
            batch_count=args.batch_count,
        )
