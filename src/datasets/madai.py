"""
Nazwa: madai.py
Opis: Główny dataset projektu.
Autor: Jan Walczak
"""
import os
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List, Tuple, Union


BoundingBox = Union[List[float], torch.Tensor]


class MADAIDataset(Dataset):
    def __init__(
        self,
        annotations_dir: str,
        image_dir: str,
        image_size=(416, 416),
        transform=None,
        target_transform=None,
    ) -> None:
        self.annotations_dir = annotations_dir
        self.img_dir = image_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else ToTensor()
        self.target_transform = target_transform
        # Get list of images and list of annotations files in given directories
        self.image_list = os.listdir(image_dir)
        self.annotations_list = os.listdir(annotations_dir)

    def __len__(self) -> int:
        return (
            len(self.image_list)
            if len(self.image_list) == len(self.annotations_list)
            else -1
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple[int, int]]:
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        anno_path = os.path.join(self.annotations_dir, self.annotations_list[idx])
        bboxes = []
        # Fetch image
        image = Image.open(img_path).convert("RGB")
        org_size = image.size
        # Perform transformation on image
        if self.transform is not None:
            image = self.transform(image)
        # Fetch data about bounding boxes from annotations file
        with open(anno_path, "r") as anno_obj:
            for line in anno_obj.readlines():
                elements = [int(e) for e in line.split(",")]
                bboxes.append(elements)
        # Apply transformation to every bounding box for a single image
        if self.target_transform is not None:
            bboxes = [self.target_transform(org_size, b) for b in bboxes]
        return image, bboxes, img_path, org_size  # type: ignore


class ResizeAndPadImage:
    """
    Responsible for resizing, centering and padding the original images to fit the given img_size.
    """

    def __init__(self, img_size, resampling=None):
        self.img_size = img_size
        self.resampling = resampling
        if self.resampling is None:
            if PIL.__version__ == "9.5.0":
                self.resampling = Image.Resampling.LANCZOS
            else:
                self.resampling = Image.LANCZOS

    def __call__(self, img: Image.Image) -> Image.Image:
        GREY = (128, 128, 128)
        ratio = img.width / img.height
        # Get the correct new size for the image
        new_size = (
            (self.img_size, int(self.img_size / ratio))
            if ratio > 1
            else (int(self.img_size * ratio), self.img_size)
        )
        # Resize the image
        img = img.resize(new_size, self.resampling)  # type: ignore
        # Center the image and pad the rest
        new = Image.new(img.mode, (self.img_size, self.img_size), GREY)
        if ratio > 1:
            new.paste(img, (0, (self.img_size - img.height) // 2))
        else:
            new.paste(img, ((self.img_size - img.width) // 2, 0))
        return new


class ResizeAndPadBoxes:
    """
    Responsible for resizing and moving the bounding boxes for them to fit the outcome image from ResizeAndPadImage.
    """

    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(
        self, org_size: Tuple[int, int], bbs: BoundingBox, inverse=False
    ) -> BoundingBox:
        ratio = org_size[0] / org_size[1]
        # Get scaled size of the image
        if ratio > 1:
            scaled_size = self.img_size, int(self.img_size / ratio)
        else:
            scaled_size = int(self.img_size * ratio), self.img_size
        # Resize and pad
        if not inverse:
            bbs = scale_bbs(bbs, org_size, scaled_size)
            bbs = center_bbs(bbs, scaled_size, (self.img_size, self.img_size), ratio)
        else:
            bbs = center_bbs(
                bbs,
                tuple([-x for x in scaled_size]),
                (-self.img_size, -self.img_size),
                ratio,
            )
            bbs = scale_bbs(bbs, scaled_size, org_size)
        return bbs


def center_bbs(
    bbox: BoundingBox,
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    ratio: float,
) -> BoundingBox:
    """
    Center the bounding box along shorter image axis.
    """
        
    if ratio > 1:
        bbox[1] += (new_size[1] - old_size[1]) // 2  # type: ignore
        bbox[3] += (new_size[1] - old_size[1]) // 2  # type: ignore
    else:
        bbox[0] += (new_size[0] - old_size[0]) // 2  # type: ignore
        bbox[2] += (new_size[0] - old_size[0]) // 2  # type: ignore
    return bbox


def scale_bbs(
    bbox: BoundingBox, old_size: Tuple[int, int], new_size: Tuple[int, int]
) -> BoundingBox:
    """
    Scale the bounding box to the new image size.
    """
    ratio_x, ratio_y = new_size[0] / old_size[0], new_size[1] / old_size[1]
    rounding_fun = torch.round if isinstance(bbox, torch.Tensor) else round
    bbox[0] = rounding_fun(bbox[0] * ratio_x)  # type: ignore
    bbox[2] = rounding_fun(bbox[2] * ratio_x)  # type: ignore
    bbox[1] = rounding_fun(bbox[1] * ratio_y)  # type: ignore
    bbox[3] = rounding_fun(bbox[3] * ratio_y)  # type: ignore
    return bbox
