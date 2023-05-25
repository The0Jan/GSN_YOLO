"""
Nazwa: primitive.py
Opis: Dataset zwracający tylko zdjęcia z folderu (bez adnotacji).
Autor: Bartłomiej Moroz
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Tuple


class PrimitiveDataset(Dataset):
    """
    A dataset that provides input images info in the same format as MADAIDataset,
    but with no annotations. Primary use case is for manual testing and predictions.
    """

    Item = Tuple[torch.Tensor, list, str, Tuple[int, int]]

    def __init__(self, image_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.transform = transform if transform is not None else ToTensor()
        self.image_list = os.listdir(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Item:
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        bboxes = []
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        if self.transform is not None:
            image = self.transform(image)
        return image, bboxes, image_path, original_size  # type: ignore
