import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from typing import Tuple
import torch


class PrimitiveDataset(Dataset):
    Item = Tuple[torch.Tensor, list, str, Tuple[int, int]]

    def __init__(self, image_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.transform = transform if transform is not None else Compose([ToTensor()])
        self.image_list = os.listdir(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Item:
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        bboxes = []
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        if self.transform is not None:
            image = self.transform(image)
        return image, bboxes, image_path, original_size