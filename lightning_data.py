import torch
import dataset
import gdown
import zipfile
import pytorch_lightning as pl
from torchvision.transforms import Compose
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, random_split


class MADAIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 16, image_size = (416, 416),
                 train_anno_dir = 'train-new/annotations', train_img_dir = 'train-new/images',
                 test_anno_dir = 'test-new/annotations', test_img_dir = 'test-new/images',
                 img_transform = None, target_transform = None) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.image_size     = image_size
        self.train_anno_dir = train_anno_dir
        self.train_img_dir  = train_img_dir
        self.test_anno_dir  = test_anno_dir
        self.test_img_dir   = test_img_dir
        
        self.g_id = "1sDqxwOeROzsfvW2d_K7O_akEyLhDcKa3"
        self.file_name = "data.zip"
        self.img_transform  = img_transform
        if img_transform is None:
            self.img_transform = self.get_img_transform()
        
        self.target_transform   = target_transform
        if target_transform is None:
            self.target_transform = self.get_target_transform() 

    def prepare_data(self):
        #pobieranie odpowiednich zipów 
        #gdown.download(id = self.g_id, output=self.file_name)
        #ziper = zipfile.ZipFile(self.file_name)
        #ziper.extractall()
        #ziper.close()
        return 0
        
    def setup(self, stage=None):
        # called on every GPU
        # use our dataset and defined transformations
        if stage == 'fit' or stage is None:
            dataset_yolo = dataset.YOLODataset(self.train_anno_dir, self.train_img_dir, 
                                               transform = self.img_transform,
                                               target_transform = self.target_transform,
                                               image_size = self.image_size)
            
            train_dataset_size = int(len(dataset_yolo) * 0.9)
            self.dataset_train, self.dataset_val  = random_split(dataset_yolo, [train_dataset_size, len(dataset_yolo) - train_dataset_size])
            
        if stage == 'test' or stage is None:
            self.dataset_test = dataset.YOLODataset(self.test_anno_dir, self.test_img_dir,
                                               transform = self.img_transform,
                                               target_transform = self.target_transform,
                                               image_size = self.image_size)
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2, collate_fn= _collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2, collate_fn= _collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2, collate_fn= _collate_fn)
    
    def get_img_transform(self):
        return Compose([Lambda(dataset.resize_with_respect), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def get_target_transform(self):
        return dataset.resize_bbs


def _collate_fn(batch):
    """_summary_
        img_tensor, img_path, org_size, target
    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    image_batch = torch.stack([elem[0] for elem in batch], 0)
    img_path_batch = [elem[2] for elem in batch]
    org_size_batch = [elem[3] for elem in batch]
    
    #annotation_batch = torch.Tensor([assign_batch_index_to_bbx(annotations, batch_index) for batch_index, annotations in enumerate([elem[1] for elem in batch])]),
    annotation_batch = torch.cat(
        [
            torch.cat(
                (
                    torch.ones(len(annotations), 1) * batch_index,
                    torch.Tensor(annotations),
                ),
                1,
            )
            for batch_index, annotations in enumerate([elem[1] for elem in batch])
        ],
        0,
    )

    return (image_batch, annotation_batch, img_path_batch, org_size_batch) 


def assign_batch_index_to_bbx(target, batch_index):
    for i, bbx in enumerate(target):
        target[i] = [batch_index, *bbx]
    return target
