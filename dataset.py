"""
Nazwa: dataset.py
Opis: Główny dataset projektu. Zawiera także funkcje do wizualizacji wyników
      (TODO: przenieść wizualizację do innego pliku)
Autor: Jan Walczak
"""
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import ToPILImage
import PIL
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
        
        # Get list of images and list of annotations files in given directories
        self.image_list = os.listdir(image_dir)
        self.annotations_list = os.listdir(annotations_dir)

    def __len__(self) -> int:
        return len(self.image_list) if len(self.image_list) == len(self.annotations_list) else -1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple[int, int]]:
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        anno_path = os.path.join(self.annotations_dir, self.annotations_list[idx])
        bboxes = []
        
        # Fetch image
        image = Image.open(img_path).convert('RGB')
        org_size = image.size
        # Perform transformation on image
        if self.transform is not None:
            image = self.transform(image)
            
        # Fetch data about bounding boxes from annotations file
        with open(anno_path, 'r') as anno_obj:
            for line in anno_obj.readlines():
                elements = [int(e) for e in line.split(",")]
                bboxes.append(elements)
        # Apply transformation to every bounding box for a single image       
        if self.target_transform is not None:
            bboxes = [self.target_transform(org_size, self.image_size, b) for b in bboxes]
        return image, bboxes, img_path, org_size,


class ResizeAndPadImage():
    """
    Responsible for resizing, centering and padding the original images to fit the given img_size.
    """
    def __init__(self, img_size, resampling=None):
        self.img_size = img_size
        self.resampling = resampling
        if self.resampling is None:
            if PIL.__version__ == '9.5.0':
                self.resampling = Image.Resampling.LANCZOS
            else:
                self.resampling = Image.LANCZOS

    def __call__(self, img: Image.Image) -> Image.Image:
        GREY = (128, 128, 128)
        ratio = img.width / img.height
        # Get the correct new size for the image
        new_size = (self.img_size, int(self.img_size / ratio)) if ratio > 1 else (int(self.img_size * ratio), self.img_size)
        # Resize the image 
        img = img.resize(new_size, self.resampling)
        # Center the image and pad the rest
        new = Image.new(img.mode, (self.img_size, self.img_size), GREY)
        if ratio > 1:
            new.paste(img, (0, (self.img_size - img.height) // 2))
        else:
            new.paste(img, ((self.img_size - img.width) // 2, 0))
        return new


class ResizeAndPadBoxes():
    """
    Responsible for resizing and moving the bounding boxes for them to fit the outcome image from ResizeAndPadImage.
    """
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, org_size, new_size, bbs) -> list:
        ratio = org_size[0]/org_size[1]
        # Get scaled size of the image (we use this for resizing )
        if ratio > 1:
            s_size = int(new_size[0]), int(new_size[1] / ratio)
        else:
            s_size = int(new_size[0] * ratio), int(new_size[1]) 
        # Scale coordinates
        bbs = scale_bbs(org_size, s_size, bbs)
        # Move coordinates (This is done, due to us moving the image with ResizeAndPadImage to the center)
        if ratio > 1:
            bbs[2] = add_cord(bbs[2],new_size[1],s_size[1])
            bbs[4] = add_cord(bbs[4],new_size[1],s_size[1])
        else:
            bbs[1] = add_cord(bbs[1],new_size[0],s_size[0])
            bbs[3] = add_cord(bbs[3],new_size[0],s_size[0])
        # Normalize (Actually is currently not used)
        #bbs = norm(bbs, new_size[0])
        return bbs

def norm(cords, img_side):
    """
    Early function used for normalizing the coordinate values for input.
    """
    for i in range(1, len(cords)):
        cords[i] = cords[i] / img_side
    return cords

def add_cord(corn, new_s, s_size):
    return int(corn +  (new_s - s_size) / 2)


def scale_bbs(org_size, new_size, bbs):
    """
    Scale the bounding boxes to the new image size.
    """
    Rx = new_size[0] / org_size[0]
    Ry = new_size[1] / org_size[1]
    bbs[1] = round(bbs[1] * Rx)
    bbs[2] = round(bbs[2] * Ry)
    bbs[3] = round(bbs[3] * Rx)
    bbs[4] = round(bbs[4] * Ry)
    return bbs


def tensor_to_image(tensor_image): 
    """
    Early function used for changing back the given transformed tensor to an image.
    """
    inv_imagenet_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1./0.229, 1./0.224, 1./0.225])
    to_pil_image = Compose([inv_imagenet_normalize, ToPILImage()])
    image = to_pil_image(tensor_image)
    return image

def inv_resize_bbs(org_size, new_size, bbs):
    """
    Function responsible for undoing the scaling and appropriate moving of coordinates done by 'ResizeAndPadBoxes'.
    Prepares the bounding boxes to be used on the original image.
    """
    ratio = new_size[0]/new_size[1]
    # Get what the scaled sized of the image was
    s_size = org_size[0], int(org_size[1] / ratio) if ratio > 1 else int(org_size[0] * ratio), org_size[1]
    # Move coordinates back
    if ratio > 1:
        bbs[2] = sub_cord(bbs[2], org_size[1], s_size[1])
        bbs[4] = sub_cord(bbs[4], org_size[1], s_size[1])
    else:
        bbs[1] = sub_cord(bbs[1], org_size[0], s_size[0])
        bbs[3] = sub_cord(bbs[3], org_size[0], s_size[0])
    # Scale coordinates back
    bbs = scale_bbs(s_size, new_size, bbs)  
    return bbs

def sub_cord(corn, new_s, s_size):
    return int(corn -  (new_s- s_size)/2)

def draw_box(image, bbx, target_class):
    """
    Draw a single bounding box with class onto a given image.
    Help and inspiration drawn from https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    """
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
    # Adding what class onto bouding box
    label = "{0}".format(label)
    text_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 0.4, 1)[0]
    corner_2 = corner_1[0] + text_size[0] + 4, corner_1[1] + text_size[1] + 4
    cv2.rectangle(image, corner_1, corner_2, color, -1)
    cv2.putText(image, label, (corner_1[0], corner_1[1] + text_size[1] + 4), cv2.FONT_ITALIC, 0.4, [255,255,255], 1)
    return image


def visualize_results(img_path, out_dir, targets):
    """
    Draw all bounding boxes with corresponding class onto a single image and saving the results.
    """
    image = Image.open(img_path).convert('RGB')
    cur_size = image.width, image.height
    image = np.array(image)
    for target in targets:
        bbx = target[0:5]
        bbx = inv_resize_bbs((416,416), cur_size, bbx)
        image = draw_box(image, bbx[1:], target[6])
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    Image.fromarray(image).save(out_path)
    return image
