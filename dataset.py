import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Lambda
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2
import numpy as np
from typing import Tuple
import torch

class YOLODataset(Dataset):
    def __init__(self, annotations_dir: str, image_dir: str, image_size=(416, 416),
                 transform=None, target_transform=None) -> None:
        self.annotations_dir = annotations_dir
        self.img_dir = image_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else Compose([ToTensor()])
        self.target_transform = target_transform
        #
        self.image_list = os.listdir(image_dir)
        self.annotations_list = os.listdir(annotations_dir)

    def __len__(self) -> int:
        return len(self.image_list) if len(self.image_list) == len(self.annotations_list) else -1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple[int, int]]:
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        anno_path = os.path.join(self.annotations_dir, self.annotations_list[idx])
        bboxes = []
        image = Image.open(img_path).convert('RGB')
        org_size = image.size
        if self.transform is not None:
            image = self.transform(image)
        with open(anno_path, 'r') as anno_obj:
            for line in anno_obj.readlines():
                elements = [int(e) for e in line.split(",")]
                bboxes.append(elements)
        if self.target_transform is not None:
            bboxes = [self.target_transform(org_size, self.image_size, b) for b in bboxes]
        return image, bboxes, img_path, org_size,


def resize_with_respect(img: Image.Image) -> Image.Image:
    IMG_SIDE = 416
    GREY = (128, 128, 128)
    ratio = img.width / img.height
    new_size = (IMG_SIDE, int(IMG_SIDE / ratio)) if ratio > 1 else (int(IMG_SIDE * ratio), IMG_SIDE)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new = Image.new(img.mode, (IMG_SIDE, IMG_SIDE), GREY)
    if ratio > 1:
        new.paste(img, (0, (IMG_SIDE - img.height) // 2))
    else:
        new.paste(img, ((IMG_SIDE - img.width) // 2, 0))
    return new


def resize_bbs(org_size, new_size, bbs):
    ratio = org_size[0]/org_size[1]
    # Get scale
    if ratio > 1:
        s_size = int(new_size[0]), int(new_size[1] / ratio)
    else:
        s_size = int(new_size[0] * ratio), int(new_size[1]) 
    # Scale coordinates
    bbs = scale_bbs(org_size, s_size, bbs)
    # Move coordinates
    if ratio > 1:
        bbs[2] = add_cord(bbs[2],new_size[1],s_size[1])
        bbs[4] = add_cord(bbs[4],new_size[1],s_size[1])
    else:
        bbs[1] = add_cord(bbs[1],new_size[0],s_size[0])
        bbs[3] = add_cord(bbs[3],new_size[0],s_size[0])
    # Normalize
    # Wyłączone do testów inv_resize
    # naah, jednak chcemy od 0 do 416
    #bbs = norm(bbs, new_size[0])
    return bbs


def norm(cords, img_side):
    for i in range(1, len(cords)):
        cords[i] = cords[i] / img_side
    return cords


def add_cord(corn, new_s, s_size):
    return int(corn +  (new_s - s_size) / 2)


def scale_bbs(org_size, new_size, bbs):
    Rx = new_size[0] / org_size[0]
    Ry = new_size[1] / org_size[1]
    bbs[1] = round(bbs[1] * Rx)
    bbs[2] = round(bbs[2] * Ry)
    bbs[3] = round(bbs[3] * Rx)
    bbs[4] = round(bbs[4] * Ry)
    return bbs


def tensor_to_image(tensor_image): 
    inv_imagenet_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1./0.229, 1./0.224, 1./0.225])
    to_pil_image = Compose([inv_imagenet_normalize, ToPILImage()])
    image = to_pil_image(tensor_image)
    return image


def inv_resize_bbs(org_size, new_size, bbs):
    ratio = new_size[0]/new_size[1]
    # Get Scale
    s_size = org_size[0], int(org_size[1] / ratio) if ratio > 1 else int(org_size[0] * ratio), org_size[1]
    # Move coordinates
    if ratio > 1:
        bbs[2] = sub_cord(bbs[2], org_size[1], s_size[1])
        bbs[4] = sub_cord(bbs[4], org_size[1], s_size[1])
    else:
        bbs[1] = sub_cord(bbs[1], org_size[0], s_size[0])
        bbs[3] = sub_cord(bbs[3], org_size[0], s_size[0])
    #Scale coordinates
    bbs = scale_bbs(s_size, new_size, bbs)  
    return bbs


def sub_cord(corn, new_s, s_size):
    return int(corn -  (new_s- s_size)/2)


def draw_box(image, bbx, target_class):
    corner_1 = tuple(bbx[0:2])
    corner_2 = tuple(bbx[2:4])
    classes = {
        0: ("aircraft", (76, 155, 78)), 
        1: ("bomber", (223, 127, 56)),
        2: ("early warning aircraft", (92, 125, 159)),
        3: ("fighter", (103,193,173)),
        4: ("military helicopter", (176, 65, 64))
    }
    #label, color = classes[int(target_class)]
    label, color = int(target_class), (255, 0, 0)
    # Drawing box on image
    cv2.rectangle(image, corner_1, corner_2, color, 2)
    # Wiriting what class
    label = "{0}".format(label)
    text_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 0.4, 1)[0]
    corner_2 = corner_1[0] + text_size[0] + 4, corner_1[1] + text_size[1] + 4
    cv2.rectangle(image, corner_1, corner_2, color, -1)
    cv2.putText(image, label, (corner_1[0], corner_1[1] + text_size[1] + 4), cv2.FONT_ITALIC, 0.4, [255,255,255], 1)
    return image


def visualize_results(img_path, targets):
    image = Image.open(img_path).convert('RGB')
    cur_size = image.width, image.height
    image = np.array(image)
    for target in targets:
        bbx = target[0:5]
        bbx = inv_resize_bbs((416,416), cur_size, bbx)
        image = draw_box(image, bbx[1:], target[6])
    Image.fromarray(image).show()
    return image
