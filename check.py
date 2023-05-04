import yolo
import darknet
from loading_weights import load_weights
import os
from torchsummary import summary
dir = 'weights_files'
file_name1 =  os.path.join(dir, 'darknet53.conv.74')
file_name2 = os.path.join(dir, 'yolov3.weights')



            
            
model = darknet.Darknet53(3)

model = yolo.YOLOv3(80, 3)

#for  n, p in model.named_parameters():
#    print(n, p.numel())
    
summary(model, input_size=(3,416,416), device='cpu')
    
#load_weights(file_name2, model)

#for int,  param in enumerate(model.named_parameters(), 0):
#    if int ==0:
#        print(param)