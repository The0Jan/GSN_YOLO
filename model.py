import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.con = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1) # Recheck if correct value for RelU
        
    def forward(self, x):
        x = self.con(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x
    

class ResUnit(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels //2, kernel_size = 1)
        self.conv2 = ConvBlock(channels// 2, channels, kernel_size = 3, padding = 1) # Nie jestem pewien jeszcze tutaj co do paddingu więc dałem wstępnie taki

    def forward(self, x):
        # Tak jest w dokumentacji więc powinno być git
        pre_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + pre_x

class ResBlock(nn.Module):
    def __init__(self, n_times, in_chan, out_chan, kernel, stride) -> None:
        super().__init__()
        self.zero_pad_conv = ConvBlock(in_chan, out_chan, kernel_size = kernel, stride = stride, padding = 0)
        self.n_resunits = nn.ModuleList()
        for _ in range(n_times):
            self.n_resunits += ResUnit(out_chan)
            
    def forward(self, x):
        for resunit in self.n_resunits:
            x = x + resunit(x)
        
        
class Darknet53(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.con = ConvBlock(in_channels, 32, kernel_size = 3, stride = 1)
        self.res_1 = ResBlock(1 , 32, 64, kernel = 3, stride = 2)
        self.res_2 = ResBlock(2 , 64, 64, kernel = 3, stride = 2)
        self.res_3 = ResBlock(8 , 128, 64, kernel = 3, stride = 2)
        self.res_4 = ResBlock(8 , 256, 64, kernel = 3, stride = 2)
        self.res_5 = ResBlock(4 , 512, 1024, kernel = 3, stride = 2)

        self.avgpool
        self.connected
        self.softmax
        

    def forward(self, x):
        x = self.con(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        return x