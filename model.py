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
import custom_dataset

class autoencoder(nn.Module):

    def auto_pad(self):
        
        return ((self.stride - ((self.img_dim[0]-self.kernel)%self.stride))//2,(self.stride - ((self.img_dim[1]-self.kernel)%self.stride))//2)
    
    def output_shape(self):

        """
        Returns the shape of the autopadded tensor when convolved
        """
        
        padding = self.auto_pad()
        return (1+((self.img_dim[0]+2*padding[0]-self.kernel)//self.stride),1+((self.img_dim[1]+2*padding[1]-self.kernel)//self.stride))

    def __init__(self, learning_rate):
        super(autoencoder, self).__init__()
       
        self.mode = 1       # 1 corresponds to target sample      

        # Architecture parameters
        maps0 = 3   # Initially 12, when last conv layer is active
        maps1 = 20      
        maps2 = 34     
        maps3 = 50     
        maps4 = 80
        dense_compression = 10
        self.neg_slope = 0.1        # slope value for leaky_relu
        self.kernel = 4
        self.stride = 2

        self.img_dim = custom_dataset.IMAGE_SIZE

        self.encoding_layer = []
        self.decoding_layer = []        
        
        pad_params = []
        for i in range(4):
            pad_params.append(self.auto_pad())
            self.img_dim = self.output_shape()

        self.conv1 = nn.Conv2d(3, maps1, self.kernel, stride=self.stride, padding = pad_params[0])        
        self.conv2 = nn.Conv2d(maps1, maps2, self.kernel, stride=self.stride, padding = pad_params[1])       
        self.conv3 = nn.Conv2d(maps2, maps3, self.kernel, stride=self.stride, padding = pad_params[2])                    
        self.conv4 = nn.Conv2d(maps3, maps4, self.kernel, stride=self.stride, padding = pad_params[3])    
        dense_neurons = maps4 * self.img_dim[0] * self.img_dim[1]
        self.dense1 = nn.Linear(dense_neurons, dense_neurons//dense_compression)
        self.dense2 = nn.Linear(dense_neurons//dense_compression, dense_neurons)              

        self.encoding_layer.append(self.conv1)      
        self.encoding_layer.append(self.conv2)      
        self.encoding_layer.append(self.conv3)                 
        self.encoding_layer.append(self.conv4)    
        self.encoding_layer.append(self.dense1)
        self.encoding_layer.append(self.dense2)
    
        # Decoder werights for mode = 0, i.e, source
        # self.dense10 = nn.Linear(dense_neurons//dense_compression, dense_neurons)
        self.deconv10 = nn.ConvTranspose2d(maps4, maps3, self.kernel, stride=self.stride, padding = pad_params[3])
        self.deconv20 = nn.ConvTranspose2d(maps3, maps2, self.kernel, stride=self.stride, padding = pad_params[2])
        self.deconv30 = nn.ConvTranspose2d(maps2, maps1, self.kernel, stride=self.stride, padding = pad_params[1])
        self.deconv40 = nn.ConvTranspose2d(maps1, maps0, self.kernel, stride=self.stride, padding = pad_params[0])  
        # self.conv50 = nn.Conv2d(maps0, 3, 3, stride=1, padding=1)

        # Decoder weights for mode = 1, i.e, target
        # self.dense11 = nn.Linear(dense_neurons//dense_compression, dense_neurons)
        self.deconv11 = nn.ConvTranspose2d(maps4, maps3, self.kernel, stride=self.stride, padding = pad_params[3])
        self.deconv21 = nn.ConvTranspose2d(maps3, maps2, self.kernel, stride=self.stride, padding = pad_params[2])
        self.deconv31 = nn.ConvTranspose2d(maps2, maps1, self.kernel, stride=self.stride, padding = pad_params[1])
        self.deconv41 = nn.ConvTranspose2d(maps1, maps0, self.kernel, stride=self.stride, padding = pad_params[0])
        # self.conv51 = nn.Conv2d(maps0, 3, 3, stride=1, padding=1)

        self.decoding_layer.append([])      
        # self.decoding_layer[-1].append(self.dense10)
        self.decoding_layer[-1].append(self.deconv10)
        self.decoding_layer[-1].append(self.deconv20)  
        self.decoding_layer[-1].append(self.deconv30)  
        self.decoding_layer[-1].append(self.deconv40)
        # self.decoding_layer[-1].append(self.conv50)

        self.decoding_layer.append([])              
        # self.decoding_layer[-1].append(self.dense11) 
        self.decoding_layer[-1].append(self.deconv11)                         
        self.decoding_layer[-1].append(self.deconv21)                                             
        self.decoding_layer[-1].append(self.deconv31)                                             
        self.decoding_layer[-1].append(self.deconv41)  
        # self.decoding_layer[-1].append(self.conv51)

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
            self.optimiser.append(torch.optim.Adam(list(self.encoder_params)+list(self.decoder_params[i]), lr=learning_rate[i], weight_decay=1e-5))

    def step(self):
        
        self.optimiser[self.mode].step()
        self.optimiser[self.mode].zero_grad()

    def encoder(self, x):

        self.x0 = x            
        # print("Initial image:",self.x0.shape)
        self.x1 = F.leaky_relu(self.encoding_layer[0](self.x0),self.neg_slope)
        # print("Conv 1:",self.x1.shape)
        self.x2 = F.leaky_relu(self.encoding_layer[1](self.x1),self.neg_slope)
        # print("Conv 2:",self.x2.shape)
        self.x3 = F.leaky_relu(self.encoding_layer[2](self.x2),self.neg_slope)
        # print("Conv 3:",self.x3.shape)
        self.x4 = F.leaky_relu(self.encoding_layer[3](self.x3),self.neg_slope)
        # print("Conv 4:",self.x4.shape)
        self.x5 = F.leaky_relu(self.encoding_layer[4](self.x4.reshape((self.x4.shape[0], self.x4.numel()//self.x4.shape[0]))),self.neg_slope)
        # print("Dense 1:",self.x5.shape)    
        self.x6 = F.leaky_relu(self.encoding_layer[5](self.x5), self.neg_slope)
        # print("Dense 2:",self.x6.shape)
        self.x6 = self.x6.reshape(self.x4.shape) 
        
        return self.x6

    def decoder(self, x):

        if(self.mode):
            
            # x = F.leaky_relu(self.decoding_layer[1][0](x), self.neg_slope)
            # x = x.reshape(self.x4.shape)
            x = F.leaky_relu(self.decoding_layer[1][0](x), self.neg_slope) # + self.x4    # skip connection (optional - mostly useless)
            # print("Deconv 1:",x.shape)
            x = F.leaky_relu(self.decoding_layer[1][1](x), self.neg_slope) # + self.x3 
            # print("Deconv 2:",x.shape)
            x = F.leaky_relu(self.decoding_layer[1][2](x), self.neg_slope) # + self.x2 
            # print("Deconv 3:",x.shape)
            x = F.leaky_relu(self.decoding_layer[1][3](x), self.neg_slope) # + self.x1 
            # print("Deconv 4:",x.shape)
            # x = F.leaky_relu(self.decoding_layer[1][4](x), self.neg_slope) 
            # print("Conv 5:",x.shape)
            x = torch.sigmoid(x)
            
        else:
            
            # x = F.leaky_relu(self.decoding_layer[0][0](x), self.neg_slope)
            # x = x.reshape(self.x4.shape)
            x = F.leaky_relu(self.decoding_layer[0][0](x), self.neg_slope) # + self.x4    # skip connection (optional - mostly useless)
            # print("Deconv 1:",x.shape)
            x = F.leaky_relu(self.decoding_layer[0][1](x), self.neg_slope) # + self.x3 
            # print("Deconv 2:",x.shape)
            x = F.leaky_relu(self.decoding_layer[0][2](x), self.neg_slope) # + self.x2 
            # print("Deconv 3:",x.shape)
            x = F.leaky_relu(self.decoding_layer[0][3](x), self.neg_slope) # + self.x1 
            # print("Deconv 4:",x.shape)
            # x = F.leaky_relu(self.decoding_layer[1][4](x), self.neg_slope) 
            # print("Conv 5:",x.shape)
            x = torch.sigmoid(x)

        return x

    def setMode(self, mode):

        self.mode = mode

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
