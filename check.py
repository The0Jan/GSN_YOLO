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
    summary(model, input_size=(3,416,416), device='cpu')
        
    load_model_parameters(path_to_yolo, model)

    train_path_anno = 'test-new/annotations'
    train_path_img =  'test-new/images'
    yolo_dataset = YOLODataset(train_path_anno, train_path_img)

    x, path, _, bboxes = yolo_dataset[0]
    x = x.unsqueeze(0)
    print(path)
    print(bboxes)
    print(x.shape)
    y = model(x)
    print(y.shape)
    z = after_party(y)
    print(z.shape)
    print(z)
