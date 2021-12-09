import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        if x.size()[1]== 1: 
            x = x.repeat(1,3,1,1)    
        return self.conv(x)

class DoubleConv_no_pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_no_pool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if x.size()[1]== 1: 
            x = x.repeat(1,3,1,1)    
        return self.conv(x)

class UNET_encoder(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self):
        super().__init__()
        width = 64
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('unit1', DoubleConv_no_pool(3, width))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit2', DoubleConv(width, width*2))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit3', DoubleConv(width*2, width*4))] 
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit4', DoubleConv(width*4, width*8))] 
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit5', DoubleConv(width*8, width*16))] 
                ))),
        ]))

    def forward(self, x):
        features = []
        x = self.root(x)
        b, c, in_size, _ = x.size()
        features.append(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]
