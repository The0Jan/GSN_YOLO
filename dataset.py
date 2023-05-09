import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2
import numpy as np

def loader(path):
    return Image.open(path).convert('RGB')

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
        image = loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        with open(anno_path, 'r') as anno_obj:
            csv_lines = anno_obj.readlines()
            
            for index, line in enumerate(csv_lines):
                elements = line.split(",")
                for i, e in enumerate(elements):
                    elements[i] = int(e)
                if(index == 0 ):
                    org_size = elements
                else:
                    bboxes.append(elements)
                    
        if self.target_transform:
            for index in range(len(bboxes)):
                bboxes[index] = self.target_transform(org_size, self.image_size, bboxes[index])
             
        return image, img_path, org_size, bboxes
    

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

###########################
check = False
if check:
    train_path_anno = 'test-new/annotations'
    train_path_img =  'test-new/images'

    image_size = (416, 416)
    img_transform = Compose([Resize(image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    yolo_dataset = YOLODataset(train_path_anno, train_path_img, transform=img_transform )

    for i in range(10): 
        print(yolo_dataset.annotations_list[i], yolo_dataset.image_list[i])
    #tensor_image, org_size, label = yolo_dataset[8]
    #image = tensor_to_image(tensor_image)
    #image.show()
    #print(org_size)
    #print(label)