"""
Nazwa: visualizing_results.py
Opis: Funkcje do wizualizacji wyników.
Autor: Jan Walczak
"""
import cv2
import numpy as np
import os
from PIL import Image
from src.datasets.madai import BoundingBox, ResizeAndPadBoxes
import torch
from typing import List, Union


def draw_box(image: np.ndarray, bbox: BoundingBox, target_class: int) -> np.ndarray:
    """
    Draw a single bounding box with class onto a given image.
    Help and inspiration drawn from https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    """
    WHITE = (255, 255, 255)
    corner_1, corner_2 = tuple(bbox[0:2]), tuple(bbox[2:4])
    classes = {
        0: ("aircraft", (76, 155, 78)),
        1: ("bomber", (223, 127, 56)),
        2: ("early warning aircraft", (92, 125, 159)),
        3: ("fighter", (103, 193, 173)),
        4: ("military helicopter", (176, 65, 64)),
    }
    label, color = classes[target_class]
    # Drawing box on image
    cv2.rectangle(image, corner_1, corner_2, color, thickness=2)
    # Adding class text box
    label = f"{label}"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_ITALIC, fontScale=0.4, thickness=1)
    corner_2 = corner_1[0] + text_size[0] + 4, corner_1[1] + text_size[1] + 4
    cv2.rectangle(image, corner_1, corner_2, color, thickness=-1)
    cv2.putText(
        image,
        label,
        (corner_1[0], corner_2[1]),
        cv2.FONT_ITALIC,
        0.4,
        WHITE,
        thickness=1,
    )
    return image


def visualize_results(
    img_path: str, out_dir: str, targets: Union[torch.Tensor, List[BoundingBox]]
) -> None:
    """
    Draw all bounding boxes with corresponding class onto a single image and write it to file.
    """
    image = Image.open(img_path).convert("RGB")
    image_size = image.width, image.height
    image = np.array(image)
    resizer = ResizeAndPadBoxes(416)
    for target in targets:
        bbox = resizer(image_size, target[:5], inverse=True)[1:]
        image = draw_box(image, bbox, int(target[6]))
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    Image.fromarray(image).save(out_path)
