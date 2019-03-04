# File: autoencoder.py
# Authors:
# Nihesh Anderson
# Date : March 4, 2019

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from torchvision import datasets, models, transforms
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.misc
import time

class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
       
        self.mode = 1       # 1 corresponds to target sample      

        self.conv1 = nn.Conv2d(3, 64, 3, stride=3, padding=0)       # 360 x 240 => 120 x 80 
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)       # 120 x 80 => 120 x 80
        self.pool2 = nn.MaxPool2d(2, stride=2)                  # 120 x 80  => 60 x 40
        self.conv3 = nn.Conv2d(64, 512, 2, stride=2, padding=0)    # 60 x 40 => 30 x 20
            
        # Decoder werights for mode = 0, i.e, source
        self.deconv10 = nn.ConvTranspose2d(512, 64, 2, stride=2)        # 30 x 20 => 60 x 40
        self.deconv20 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 60 x 40 => 120 x 80
        self.deconv30 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)  # 120 x 80 => 120 x 80
        self.deconv40 = nn.ConvTranspose2d(64, 3, 3, stride=3, padding=0)  # 120 x 80 => 360 x 240
            
        # Decoder weights for mode = 1, i.e, target
        self.deconv11 = nn.ConvTranspose2d(512, 64, 2, stride=2)        # 30 x 20 => 60 x 40
        self.deconv21 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 60 x 40 => 120 x 80
        self.deconv31 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)  # 120 x 80 => 120 x 80
        self.deconv41 = nn.ConvTranspose2d(64, 3, 3, stride=3, padding=0)  # 120 x 80 => 360 x 240
    
    def encoder(self, x):

        self.x0 = x             # Removed blurring
        self.x1 = F.relu(self.conv1(self.x0))
        self.x2 = F.relu(self.conv2(self.x1))
        self.x3 = self.pool2(self.x2)
        self.x4 = F.relu(self.conv3(self.x3))

        return self.x4

    def decoder(self, x):

        if(self.mode):
            x = F.relu(self.deconv11(x)) # + self.x3    # skip connection (optional)
            x = F.relu(self.deconv21(x)) # + self.x2 
            x = F.relu(self.deconv31(x)) # + self.x1 
            x = torch.tanh(self.deconv41(x))
        else:
            x = F.relu(self.deconv10(x)) # + self.x3    # skip connection (optional)
            x = F.relu(self.deconv20(x)) # + self.x2 
            x = F.relu(self.deconv30(x)) # + self.x1 
            x = torch.tanh(self.deconv40(x))

        return x

    def setMode(self, mode):

        self.mode = mode

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
