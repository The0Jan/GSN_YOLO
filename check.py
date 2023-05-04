import yolo
import darknet
from loading_weights import load_model_parameters
import os
from torchsummary import summary


if __name__ == "__main__":
    WEIGHTS_DIR = 'weights'
    path_to_darknet = os.path.join(WEIGHTS_DIR, 'darknet53.conv.74')
    path_to_yolo = os.path.join(WEIGHTS_DIR, 'yolov3.weights')

    model = darknet.Darknet53(3)
    summary(model, input_size=(3,416,416), device='cpu')
    #model = yolo.YOLOv3(80, 3)
    #summary(model, input_size=(3,416,416), device='cpu')

    #for n, p in model.named_parameters():
    #    print(n, p.numel())
        
    load_model_parameters(path_to_darknet, model)

    #for i, param in enumerate(model.named_parameters(), 0):
    #    if i == 0:
    #        print(param)
