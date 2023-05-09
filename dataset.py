import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.transforms import ToPILImage
from PIL import Image

def loader(path):
    return Image.open(path).convert('RGB')

class YOLODataset(Dataset):
    def __init__(
                self,
                annotations_dir, 
                image_dir, 
                transform=None, 
                target_transform=None
                ):
        
        self.img_dir = image_dir
        self.annotations_dir = annotations_dir
        
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
            bboxes = self.target_transform(bboxes)  
        return image, org_size, bboxes
    



def tensor_to_image(tensor_image): 
    inv_imagenet_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1./0.229, 1./0.224, 1./0.225])
    to_pil_image = Compose([inv_imagenet_normalize, ToPILImage()])
    image = to_pil_image(tensor_image)
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