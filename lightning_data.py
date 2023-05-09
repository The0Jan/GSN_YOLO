import torch
import dataset
import pytorch_lightning as pl
from torchvision.transforms import Compose
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split

class MadaiModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32, 
                 image_size = (416,416),
                 train_anno_dir = 'train-new/annotations', 
                 train_img_dir = 'train-new/images',
                 test_anno_dir = 'test-new/annotations',
                 test_img_dir = 'test-new/images',
                 img_transform = None,
                 target_transform = None):
        super().__init__()
        self.batch_size     = batch_size
        self.image_size     = image_size
        self.train_anno_dir = train_anno_dir
        self.train_img_dir  = train_img_dir
        self.test_anno_dir  = test_anno_dir
        self.test_img_dir   = test_img_dir
        
        self.img_transform  = img_transform
        if img_transform is None:
            self.img_transform = self.get_img_transform()
        
        self.target_transform   = target_transform
        if target_transform is None:
            self.target_transform = self.get_target_transform() 

    def prepare_data(self):
        #pobieranie odpowiednich zipów 
        return 0

    def setup(self, stage=None):
        # called on every GPU
        # use our dataset and defined transformations
        if stage == 'fit' or stage is None:
            dataset_yolo = dataset.YOLODataset(self.train_anno_dir, self.train_img_dir, 
                                               transform = self.img_transform,
                                               target_transform = self.target_transform)
            
            train_dataset_size = int(len(dataset_yolo) * 0.9)
            self.dataset_train, self.dataset_val  = random_split(dataset_yolo, [train_dataset_size, len(dataset_yolo) - train_dataset_size])
            
        if stage == 'test' or stage is None:
            self.dataset_test = dataset.YOLODataset(self.test_anno_dir, self.test_img_dir,
                                               transform = self.img_transform,
                                               target_transform = self.target_transform)
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    def get_img_transform(self):
        return Compose([Resize(self.image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def get_target_transform(self):
        return None
    
    
if __name__ == "__main__":
    dm = MadaiModule()
    dm.setup()
    
    print(dm.dataset_train[0])
    img = dataset.tensor_to_image(dm.dataset_train[0][0])
    img.show()
    dm.dataset_val