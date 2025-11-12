import torch
import torch.nn as nn
from backbone.radar.def_conv import DeformableConv2d

class RadarConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RadarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #Average pooling layer
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        #Deformable convolution layer
        self.deform_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding= 3 // 2
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.deform_conv(x)
        return x