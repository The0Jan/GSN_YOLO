import torch
import dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

class MadaiModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.batch_size = 16

    def prepare_data(self):
        # download only
        #torchvision.datasets.FashionMNIST(
        #    root = './data/FashionMNIST',
        #    train = True,
        #    download = True,
        #    transform = self.get_transform()
        #)
        #torchvision.datasets.FashionMNIST(
        #    root = './data/FashionMNIST',
        #    train = False,
        #    download = True,
        #    transform = self.get_transform()
        #)
        print("yes")

    def setup(self, stage=None):
        # called on every GPU
        # use our dataset and defined transformations
        dataset_yolo = dataset.YOLODataset('train-new/annotations', 'train-new/images')
        indices = torch.randperm(len(dataset)).tolist()
        self.dataset_train = Subset(dataset_yolo, indices[-10000:])
        self.dataset_val = Subset(dataset_yolo, indices[-10000:])

        self.dataset_test = dataset.YOLODataset('test-new/annotations', 'test-new/images')
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    def get_transform(self):
        return Compose([])