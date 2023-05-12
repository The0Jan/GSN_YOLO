import torch
import pytorch_lightning as pl
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import dataset
from typing import List, Tuple
from primitive_dataset import PrimitiveDataset


class PrimitiveDataModule(pl.LightningDataModule):
    Batch = Tuple[List[torch.Tensor], List[None], List[str], List[Tuple[int, int]]]

    def __init__(self, anno_dir: str, img_dir: str,
                 batch_size=16, num_workers=2,
                 img_transform=None) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.anno_dir       = anno_dir
        self.img_dir        = img_dir
        self.num_workers    = num_workers
        self.img_transform  = self._get_img_transform() if img_transform is None else img_transform

    def prepare_data(self) -> None:
        return

    def setup(self, stage=None) -> None:
        self.dataset = PrimitiveDataset(self.img_dir, transform=self.img_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def predict_dataloader(self, shuffle=False) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def _get_img_transform(self) -> Compose:
        return Compose([dataset.ResizeAndPadImage(416), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def _collate_fn(self, batch: PrimitiveDataset.Item) -> Batch:
        image_batch = torch.stack([elem[0] for elem in batch], dim=0)
        img_path_batch = [elem[2] for elem in batch]
        org_size_batch = [elem[3] for elem in batch]
        annotation_batch = [None for elem in batch]
        return (image_batch, annotation_batch, img_path_batch, org_size_batch) 