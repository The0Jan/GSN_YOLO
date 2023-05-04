import numpy as np
import torch.nn as nn
import torch
import yolo as yolo
from typing import List, Tuple


def load_attrs(module: nn.Module, input_file, count: int, attrs: List[Tuple[str, bool]]):
    for name, use_data in attrs:
        attr = getattr(module, name)
        if use_data:
            attr_data = attr.data
        else:
            attr_data = attr
        #Cast the loaded attributes into dims of model weights & copy the data to model.
        attr_data.copy_(torch.from_numpy(np.fromfile(input_file, dtype=np.float32, count=count)).view_as(attr_data))
        setattr(module, name, attr)


def load_model_parameters(weight_file_name: str, model: nn.Module):
    with open(weight_file_name, 'rb') as weight_file:
        # Read header
        header = np.fromfile(weight_file, dtype=np.int32, count=5)
        _, _, _, seen, _ = torch.from_numpy(header)
        module_list = [module for module in model.modules()]
        # Load parameters into each layer
        for i in range(len(module_list)):
            if(module_list[i]._get_name() == 'Conv2d'):
                conv = module_list[i]
                # If there is a batch normalization layer next, load its parameters
                # otherwise load bias of this convolutional layer
                if (module_list[i+1]._get_name() == 'BatchNorm2d'):
                    bn = module_list[i+1]
                    attrs = [('bias', True), ('weight', True), ('running_mean', False), ('running_var', False)]
                    load_attrs(bn, weight_file, bn.bias.numel(), attrs)
                else:
                    attrs = [('bias', True)]
                    load_attrs(conv, weight_file, conv.bias.numel(), attrs)
                # Load weights of a convolutional layer   
                attrs = [('weight', True)]
                load_attrs(conv, weight_file, conv.weight.numel(), attrs)
