import loading_weights
import darknet
from torch import nn
from torchsummary import summary
import numpy as np
import torch.nn as nn
import torch
import yolo as yolo
from typing import List, Tuple

def load_weights(weight_file_name, model:nn.Module):
    
    with open(weight_file_name, 'rb') as weight_file:
    
    
        header = np.fromfile(weight_file, dtype =np.int32, count=5)
        # Chyba nie korzystamy z tych danych w żaden przydtany sposób
        self_header = torch.from_numpy(header)
        self_seen = self_header[3]

        weights = np.fromfile(weight_file, dtype = np.float32)
        ptr = 0
        module_list = [module for module in model.modules()]

        for i in range(len(module_list)):
            #print(module_list[i]._get_name())
            if(module_list[i]._get_name() == 'Conv2d'):
                conv = module_list[i]
                #print(conv.__sizeof__())
                try:
                    value = module_list[i+1]._get_name() == 'BatchNorm2d'
                except:
                    value = False
                if (value):
                    #print('yes')

                    bn = module_list[i+1]
                
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)                    
                else:
                    #print('no')
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                #print(num_weights)
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                #print(ptr, len(weights))
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights) 
                
                print(ptr, len(weights))  


from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms

def pad(t: Tensor):
    _, h, w = t.shape
    m = max(h, w)
    dw = m - w
    lp = dw // 2
    rp = dw - lp
    dh = m - h
    tp = dh // 2
    bp = dh - tp
    t = F.pad(t, (lp, rp, tp, bp))
    return t

def resize(t: Tensor):
    t = F.interpolate(t.unsqueeze(0), (416, 416), mode="nearest").squeeze()
    print(t[1, 200, 200])
    return t

transform = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Lambda(pad),
        transforms.Lambda(resize),
    ]
)

img = Image.open("n01630670_common_newt.JPEG").convert("RGB")
img = transform(img).cuda().unsqueeze(0)


class DarknetTest(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.darknet = darknet.Darknet53(3)
        self.global_avg_pool = nn.AvgPool2d(13)
        self.fc = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):
        out = self.darknet(x)
        out = self.global_avg_pool(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out


dark = DarknetTest(1000)
dark = dark.cuda()

load_weights('weights/darknet53.weights', dark)
summary(dark, input_size=(3,416,416))

res = dark.forward(img)
print(res.argmax())