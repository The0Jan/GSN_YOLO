import yolo
import darknet
from loading_weights import load_model_parameters
import os
from torchsummary import summary

from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataset import YOLODataset
from nms import after_party

if __name__ == "__main__":
    WEIGHTS_DIR = 'weights'
    path_to_darknet = os.path.join(WEIGHTS_DIR, 'darknet53.conv.74')
    path_to_yolo = os.path.join(WEIGHTS_DIR, 'yolov3.weights')

    model = yolo.YOLOv3(80, 3)

    load_model_parameters(path_to_yolo, model)

    train_path_anno = 'test-new/annotations'
    train_path_img = 'test-MADAI/aircraft'
    yolo_dataset = YOLODataset(train_path_anno, train_path_img)

    x, path, _, bboxes = yolo_dataset[0]
    x = x.unsqueeze(0)
    print(path)
    print(bboxes)
    y, loss = model(x, None)
    z = after_party(y)
    print(z)
