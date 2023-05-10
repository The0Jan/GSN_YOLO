import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Lambda
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2
import numpy as np

class YOLODataset(Dataset):
    def __init__(
                self,
                annotations_dir, 
                image_dir, 
                image_size=(416, 416),
                transform=None, 
                target_transform=None,
                ):
        
        self.img_dir = image_dir
        self.annotations_dir = annotations_dir
        self.image_size = image_size
        
        self.image_list = os.listdir(image_dir)
        self.annotations_list = os.listdir(annotations_dir)

        if transform is None:
            self.transform = Compose([
                Resize(self.image_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if len(self.image_list) == len(self.annotations_list):
            return len(self.image_list)
        else:
            return -1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        anno_path = os.path.join(self.annotations_dir, self.annotations_list[idx])
        bboxes = []
        image = Image.open(img_path).convert('RGB')
        org_size = image.size
        
        if self.transform:
            image = self.transform(image)
        
        with open(anno_path, 'r') as anno_obj:
            csv_lines = anno_obj.readlines()
            
            for index, line in enumerate(csv_lines):
                elements = line.split(",")
                for i, e in enumerate(elements):
                    elements[i] = int(e)
                bboxes.append(elements)
                    
        if self.target_transform:
            for index in range(len(bboxes)):
                bboxes[index] = self.target_transform(org_size, self.image_size, bboxes[index])
             
        return image, img_path, org_size, bboxes
    
def resize_with_respect(img: Image.Image) -> Image.Image:
    IMG_SIDE = 416
    GREY = (128, 128, 128)
    ratio = img.width / img.height
    if ratio > 1:
        new_size = int(IMG_SIDE), int(IMG_SIDE / ratio)
    else:
        new_size = int(IMG_SIDE * ratio), int(IMG_SIDE)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    new = Image.new(img.mode, (IMG_SIDE, IMG_SIDE), GREY)
    if ratio > 1:
        new.paste(img, (0, int((IMG_SIDE -img.height)/2)))
    else:
        new.paste(img, (int((IMG_SIDE -img.width)/2), 0))
    return new

def  resize_bbs(org_size, new_size, bbs):
    ratio = org_size[0]/org_size[1]
    print(bbs)
    
    if ratio > 1:
        s_size = int(new_size[0]), int(new_size[1] / ratio)
    else:
        s_size = int(new_size[0] * ratio), int(new_size[1])
        
    bbs = scale_bbs(org_size, s_size, bbs)
    if ratio > 1:
        bbs[2] = add_cord(bbs[2],new_size[1],s_size[1])
        bbs[4] = add_cord(bbs[4],new_size[1],s_size[1])
    else:
        bbs[1] = add_cord(bbs[1],new_size[0],s_size[0])
        bbs[3] = add_cord(bbs[3],new_size[0],s_size[0])
    return bbs

def add_cord(corn, new_s, s_size):
    return int(corn +  (new_s- s_size)/2)

def  scale_bbs(org_size, new_size, bbs):
    Rx = new_size[0]/org_size[0]
    Ry = new_size[1]/org_size[1]
    bbs[1] = round(bbs[1]*Rx)
    bbs[2] = round(bbs[2]*Ry)
    bbs[3] = round(bbs[3]*Rx)
    bbs[4] = round(bbs[4]*Ry)
    return bbs

def tensor_to_image(tensor_image): 
    inv_imagenet_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1./0.229, 1./0.224, 1./0.225])
    to_pil_image = Compose([inv_imagenet_normalize, ToPILImage()])
    image = to_pil_image(tensor_image)
    return image

def draw_box(image, target):
    target_class = target[0]
    corner_1 = tuple(target[1:3])
    corner_2 = tuple(target[3:5])
    
    classes = {
        0:("aircraft",[76,155,78]), 
        1:("bomber",[223,127,56]),
        2:("early warning aircraft",[92,125,159]),
        3:("fighter",[103,193,173]),
        4:("military helicopter",[176,65,64])
    }
    
    label, color = classes[target_class]
    
    # Drawing box on image
    cv2.rectangle(image, corner_1, corner_2, color, 2)

    # Wiriting what class
    label = "{0}".format(label)
    text_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 1, 1)[0]
    corner_2 = corner_1[0] + text_size[0] + 4, corner_1[1] + text_size[1] + 4
    cv2.rectangle(image, corner_1, corner_2, color, -1)
    cv2.putText(image, label, (corner_1[0], corner_1[1] + text_size[1] + 4), cv2.FONT_ITALIC, 1, [255,255,255], 1)
    return image

def visualize_results(image_tensor, targets):
    image = np.array(tensor_to_image(image_tensor))
    
    for target in targets:
        image = draw_box(image, target)
    
    Image.fromarray(image).show()
    return image
