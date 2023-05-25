"""
Nazwa: madai.py
Opis: Główny DataModule używany do trenowania i testowania.
Autor: Jan Walczak
"""
import gdown
import os
import pytorch_lightning as pl
from src.datasets.madai import MADAIDataset, ResizeAndPadBoxes, ResizeAndPadImage
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
import zipfile


class MADAIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, image_size=(416, 416), num_workers=2,
                 train_anno_dir='train-new/annotations', train_img_dir='train-new/images',
                 test_anno_dir='test-new/annotations', test_img_dir='test-new/images',
                 img_transform=None, target_transform=None) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.image_size     = image_size

        # Directories for test and train data
        self.train_anno_dir = train_anno_dir
        self.train_img_dir  = train_img_dir
        self.test_anno_dir  = test_anno_dir
        self.test_img_dir   = test_img_dir
        
        self.num_workers    = num_workers
        self.data_gid = "1sDqxwOeROzsfvW2d_K7O_akEyLhDcKa3"
        self.data_file = "data.zip"

        # Init default transforms for images if none were given
        self.img_transform  = img_transform
        if img_transform is None:
            self.img_transform = self.get_img_transform()
        
        # Init default transforms for targets if none were given
        self.target_transform   = target_transform
        if target_transform is None:
            self.target_transform = self.get_target_transform() 

    def prepare_data(self):
        # Download dataset
        if not os.path.isfile(self.data_file):
            gdown.download(id=self.data_gid, output=self.data_file)
        if not os.path.isdir(self.train_img_dir):
            ziper = zipfile.ZipFile(self.data_file)
            ziper.extractall()
            ziper.close()
        return 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset_yolo = MADAIDataset(self.train_anno_dir, self.train_img_dir, 
                                        transform = self.img_transform,
                                        target_transform = self.target_transform,
                                        image_size = self.image_size)
            train_dataset_size = int(len(dataset_yolo) * 0.9)
            self.dataset_train, self.dataset_val  = random_split(dataset_yolo, [train_dataset_size, len(dataset_yolo) - train_dataset_size])
        if stage == 'test' or stage is None:
            self.dataset_test = MADAIDataset(self.test_anno_dir, self.test_img_dir,
                                             transform = self.img_transform,
                                             target_transform = self.target_transform,
                                             image_size = self.image_size)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def get_img_transform(self):
        return Compose([ResizeAndPadImage(416), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_target_transform(self):
        return ResizeAndPadBoxes(416)

    def _collate_fn(self, batch):
        """
        Customized collate function for additional annotation processing, where the batch index is being added to an annotation.
        """
        image_batch = torch.stack([elem[0] for elem in batch], 0)
        img_path_batch = [elem[2] for elem in batch]
        org_size_batch = [elem[3] for elem in batch]

        annotation_batch = torch.cat(
            [
                torch.cat([torch.ones(len(annotations), 1) * batch_index, torch.Tensor(annotations)], dim=1)
                for batch_index, annotations in enumerate([elem[1] for elem in batch])
            ],
            dim=0,
        )
        return image_batch, annotation_batch, img_path_batch, org_size_batch
