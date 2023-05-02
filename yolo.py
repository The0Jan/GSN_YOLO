import darknet
import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, channels, n_times = 3) -> None:
        super().__init__()
        self.n_conv = nn.ModuleList()
        for _ in range(n_times):
            self.n_conv += [nn.Sequential(
                darknet.Convolutional(channels, channels//2, kernel_size = 1),
                darknet.Convolutional(channels// 2, channels, kernel_size = 3, padding = 1)
            )]
        
    def forward(self, x):
        for conv in self.n_conv:
            x = conv(x)       
        return x     

class Router(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size = 1, scale_factor = 2) -> None:
        super().__init__()
        self.conv = darknet.Convolutional(in_chan, out_chan, kernel_size = kernel_size)
        self.upsample = nn.Upsample( scale_factor = scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class YOLOv3(nn.Module):
    def __init__(self, num_classes, in_channels = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        ### Backbone
        self.darknet = darknet.Darknet53(in_channels)
        
        ### Scale 3 - for detecing big objects
        self.conv_block_0 = ConvolutionalBlock(1024)
        self.conv_0  = darknet.Convolutional(1024, 3*(4+1+self.num_classes), kernel_size = 1)
        
        ### Scale 2 - for detecting medium objects
        self.conv_and_upsample_1 = Router(1024, 256)
        
        self.conv_block_1 = ConvolutionalBlock(512)
        self.conv_1 = darknet.Convolutional(512, 3*(4+1+self.num_classes), kernel_size = 1)
        
        ### Scale 1 - for detecting big objects
        self.conv_and_upsample_2 = Router(512, 128)
        
        self.conv_block_2 = ConvolutionalBlock(256)
        self.conv_2 = darknet.Convolutional(256, 3*(4+1+self.num_classes), kernel_size = 1)
        
    def forward(self, x):
        results = []
        
        x = self.darknet(x)
        
        ### 1 Tensor
        x = self.conv_block_0(x)
        results.append(self.conv_0(x))
        
        
        ### 2 Tensor
        x = self.conv_and_upsample_1(x)
        
        x = torch.cat([x, self.darknet.route_con_1], dim=1)
        
        x = self.conv_block_1(x)
        results.append(self.conv_1(x))
        
        ### 3 Tensor
        x = self.conv_and_upsample_2(x)
        
        x = torch.cat([x, self.darknet.route_con_2], dim=1)
        
        x = self.conv_block_2(x)
        results.append(self.conv_2(x))
        
        return results
        
        
        
        
        