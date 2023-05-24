"""
Nazwa: loading_weights.py
Opis: Ładowanie wag z pliku do modelu.
Autor: Jan Walczak (alpha), Bartłomiej Moroz (final)
"""
import numpy as np
import torch.nn as nn
import torch


def load_params(input_file, params: list):
    """
    Lazily load data from input file into provided params.
    """
    t = 0
    for param in params:
        #Cast the loaded parameters into dims of model weights & copy the data to model.
        param.data.copy_(torch.from_numpy(np.fromfile(input_file, dtype=np.float32, count=param.numel())).view_as(param))
        t += param.numel()
    return t


def load_model_parameters(weight_file_name: str, model: nn.Module):
    """
    Load all parameters of input model from a weights file _specifically_ in P.J. Redmon's binary format.
    """
    with torch.no_grad():
        with open(weight_file_name, 'rb') as weight_file:
            # Read header
            header = np.fromfile(weight_file, dtype=np.int32, count=5)
            # TODO: Remove this line, we don't really care about this.
            _, _, _, seen, _ = torch.from_numpy(header)
            module_list = [module for module in model.modules()]
            # Load parameters into each layer
            for i in range(len(module_list)):
                if(module_list[i]._get_name() == 'Conv2d'):
                    conv = module_list[i]
                    # If there is a batch normalization layer next, load its parameters
                    # otherwise load bias of this convolutional layer
                    if (i + 1 < len(module_list) and module_list[i+1]._get_name() == 'BatchNorm2d'):
                        bn = module_list[i+1]
                        attrs = [bn.bias, bn.weight, bn.running_mean, bn.running_var]
                        load_params(weight_file, attrs)
                    else:
                        attrs = [conv.bias]
                        load_params(weight_file, attrs)
                    # Load weights of a convolutional layer   
                    attrs = [conv.weight]
                    load_params(weight_file, attrs)
