from operator import index
from unittest import result

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm


model_res = tv.models.resnet.resnet34()
class ResNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv0 = nn.Conv2d(nc,nc, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(nc)
        self.act = nn.LeakyReLU
        self.conv1 = nn.Conv2d(nc,nc, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(nc)        
        
        
    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)   
        
        
        return self.act(x + out)
    
    
class ResTruck(nn.Module):
    def __init__(self, nc, num_blocks):
        super().__init__()
        self.truck = []
        for i in range(num_blocks):
            self.truck += [ResNet(nc)]
        self.truck = nn.Sequential(*self.truck)
        
    def forward(self, x):
        return self.truck(x)
    
    
    
    
class ResNetBlock(nn.Module):
    def __init__(self, in_nc, nc, out_nc):
        super().__init__()
        self.conv0 = nn.Conv2d(in_nc,nc, kernel_size=7, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.layer1 = ResTruck(nc, 3)
        self.conv1 = nn.Conv2d(nc,nc*2,kernel_size=3,padding=1, stride=2)
        self.layer2 = ResTruck(2*nc, 4)
        self.conv2 = nn.Conv2d(nc*2,nc*4,kernel_size=3,padding=1, stride=2)
        self.layer3 = ResTruck(4*nc, 3)
        self.conv3 = nn.Conv2d(nc*4,nc*6,kernel_size=3,padding=1, stride=2)
        self.layer4 = ResTruck(6*nc, 3)
         
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(6*nc, 1000)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = self.linear(out)
        
        return out
    
psevdo_Rs_module = ResNetBlock(3, 32, 2)
count_parametrs = sum(p.numel() for p in psevdo_Rs_module.parameters())
print(count_parametrs)