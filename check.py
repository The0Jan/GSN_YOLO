import yolo
from loading_weights import load_model_parameters
import os
from torchsummary import summary

from torchvision.transforms import Compose, Lambda, ToTensor, Normalize
from dataset import YOLODataset
from nms import after_party
import dataset

from primitive_dataloader import PrimitiveDataModule


def get_img_transform():
    return Compose([Lambda(dataset.resize_with_respect), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_target_transform():
    return dataset.resize_bbs

if __name__ == "__main__":
    WEIGHTS_DIR = 'weights'
    path_to_darknet = os.path.join(WEIGHTS_DIR, 'darknet53.conv.74')
    path_to_yolo = os.path.join(WEIGHTS_DIR, 'yolov3.weights')

    model = yolo.YOLOv3(80, 3)

    load_model_parameters(path_to_yolo, model)

    #train_path_anno = 'test-new/annotations'
    #train_path_img = 'test-val2017'
    #yolo_dataset = YOLODataset(train_path_anno, train_path_img, transform=get_img_transform(), target_transform=get_target_transform())

    data_model = PrimitiveDataModule(None, 'test-val2017', batch_size=1, num_workers=4, img_transform=get_img_transform())
    data_model.setup()

    for i, batch in enumerate(data_model.predict_dataloader()):
        if i == 0:
            x, _, path, _ = batch
            x, path = x[0], path[0]
            x = x.unsqueeze(0)
            print(path)
            y, loss = model(x, None)
            z = after_party(y)
            print(z)
            dataset.visualize_results(path, z.tolist())
