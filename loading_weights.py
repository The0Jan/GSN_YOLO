import numpy as np
import torch.nn as nn
import torch
import darknet as dark
import yolo as yolo


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
                if (module_list[i+1]._get_name() == 'BatchNorm2d'):
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