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

    def __init__(self, learning_rate):
        super(autoencoder, self).__init__()
       
        self.mode = 1       # 1 corresponds to target sample      

        maps1 = 10
        maps2 = 12
        maps3 = 50

        self.encoding_layer = []
        self.decoding_layer = []        
        
        self.conv1 = nn.Conv2d(3, maps1, 3, stride=3, padding=0)       # 360 x 240 => 120 x 80 
        self.conv2 = nn.Conv2d(maps1, maps2, 3, stride=1, padding=1)       # 120 x 80 => 120 x 80
        self.pool2 = nn.MaxPool2d(2, stride=2)                  # 120 x 80  => 60 x 40
        self.conv3 = nn.Conv2d(maps2, maps3, 2, stride=2, padding=0)    # 60 x 40 => 30 x 20

        self.encoding_layer.append(self.conv1)      
        self.encoding_layer.append(self.conv2)      
        self.encoding_layer.append(self.pool2)                 
        self.encoding_layer.append(self.conv3)    
            
        # Decoder werights for mode = 0, i.e, source
        self.deconv10 = nn.ConvTranspose2d(maps3, maps2, 2, stride=2)        # 30 x 20 => 60 x 40
        self.deconv20 = nn.ConvTranspose2d(maps2, maps2, 2, stride=2, padding=0)  # 60 x 40 => 120 x 80
        self.deconv30 = nn.ConvTranspose2d(maps2, maps1, 3, stride=1, padding=1)  # 120 x 80 => 120 x 80
        self.deconv40 = nn.ConvTranspose2d(maps1, 3, 3, stride=3, padding=0)  # 120 x 80 => 360 x 240
            
        # Decoder weights for mode = 1, i.e, target
        self.deconv11 = nn.ConvTranspose2d(maps3, maps2, 2, stride=2)        # 30 x 20 => 60 x 40
        self.deconv21 = nn.ConvTranspose2d(maps2, maps2, 2, stride=2, padding=0)  # 60 x 40 => 120 x 80
        self.deconv31 = nn.ConvTranspose2d(maps2, maps1, 3, stride=1, padding=1)  # 120 x 80 => 120 x 80
        self.deconv41 = nn.ConvTranspose2d(maps1, 3, 3, stride=3, padding=0)  # 120 x 80 => 360 x 240

        self.decoding_layer.append([])
        self.decoding_layer[-1].append(self.deconv10)        
        self.decoding_layer[-1].append(self.deconv20)  
        self.decoding_layer[-1].append(self.deconv30)  
        self.decoding_layer[-1].append(self.deconv40)

        self.decoding_layer.append([])
        self.decoding_layer[-1].append(self.deconv11)                                        
        self.decoding_layer[-1].append(self.deconv21)                                             
        self.decoding_layer[-1].append(self.deconv31)                                             
        self.decoding_layer[-1].append(self.deconv41)  
            
        self.encoder_params = list([])
        for val in self.encoding_layer:
            self.encoder_params = self.encoder_params + list(val.parameters())
        
        self.decoder_params = []
        for i in range(2):
            self.decoder_params.append(list([]))
            for val in self.decoding_layer[i]:
                self.decoder_params[-1] = self.decoder_params[-1] + list(val.parameters())
       
        self.optimiser = []
        for i in range(2):
            self.optimiser.append(torch.optim.Adam(list(self.encoder_params)+list(self.decoder_params[i]), lr=learning_rate, weight_decay=1e-5))

    def step(self):
        
        self.optimiser[self.mode].step()
        self.optimiser[self.mode].zero_grad()

    def encoder(self, x):

        self.x0 = x             # Removed blurring
        self.x1 = F.relu(self.encoding_layer[0](self.x0))
        self.x2 = F.relu(self.encoding_layer[1](self.x1))
        self.x3 = self.encoding_layer[2](self.x2)
        self.x4 = F.relu(self.encoding_layer[3](self.x3))

        return self.x4

    def decoder(self, x):

        if(self.mode):

            x = F.relu(self.decoding_layer[1][0](x)) # + self.x3    # skip connection (optional)
            x = F.relu(self.decoding_layer[1][1](x)) # + self.x2 
            x = F.relu(self.decoding_layer[1][2](x)) # + self.x1 
            x = torch.tanh(self.decoding_layer[1][3](x))
            
        else:

            x = F.relu(self.decoding_layer[0][0](x)) # + self.x3    # skip connection (optional)
            x = F.relu(self.decoding_layer[0][1](x)) # + self.x2 
            x = F.relu(self.decoding_layer[0][2](x)) # + self.x1 
            x = torch.tanh(self.decoding_layer[0][3](x))            

        return x

    def setMode(self, mode):

        self.mode = mode

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
