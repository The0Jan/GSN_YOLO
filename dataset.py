import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.transforms import ToPILImage

class YOLODataset(Dataset):
    def __init__(
                self,
                annotations_dir, 
                image_dir, 
                image_size = 416,
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
        
        image = read_image(img_path)
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
    to_pil_image = Compose([ ToPILImage()])
    image = to_pil_image(tensor_image)
    return image


###########################
train_path_anno = 'test-new/Annotations'
train_path_img =  'test-new/Images'


image_size = (416, 416)
example_transform = Compose([Resize(image_size)])


yolo_dataset = YOLODataset(train_path_anno, train_path_img )


tensor_image, org_size, label = yolo_dataset[1]
image = tensor_to_image(tensor_image)
image.show()
print(org_size)
print(label)