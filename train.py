# File: autoencoder.py
# Authors:
# Nihesh Anderson
# Date : Jan 16, 2019

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
from model import autoencoder
from custom_dataset import CustomDataset
import random
import os

os.system("rm -rf dc_img")

def to_img(x):
    
    x = np.rollaxis(x, 0, 3)
    return x

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

os.system("rm -rf ./model")
os.mkdir("./model")

num_epochs = 40000
batch_size = 64
learning_rate = [1e-3, 1e-3]
OUTPUT_SAVE_RATE = 20       # Output is written to dc_img once in these many epochs
MODEL_SAVE_RATE = 200

data_dir = "./data/"

dataset = []
for i in range(2):
    dataset.append(CustomDataset(data_dir, i))

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size,
                                             shuffle=True, num_workers = 12)

              for x in range(2)}

dataset_sizes = {x: len(dataloaders[x]) for x in range(2)}

model = autoencoder(learning_rate).cuda()
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for actor in range(2):

        cum_loss = []
        tot = 0
        model.setMode(actor)
        
        for warped, original in dataloaders[actor]:
            img = warped.float()
            img = Variable(img).cuda()

            original = original.float()
            original = Variable(original).cuda()
                    
            # Forward pass
            output = model(img)
            
            loss = criterion(output, original)
            cum_loss.append(loss.item()*(warped.size()[0]))
            tot+=warped.size()[0]
            
            # Backpropagation
            loss.backward()
            model.step()        # Invokes optimiser.step() and zero_grad() internally

        print("ACTOR: "+str(actor) + ' Epoch ['+str(epoch+1)+'/'+str(num_epochs)+'], loss:'+str(sum(cum_loss)/tot))

    if((epoch+1)%OUTPUT_SAVE_RATE == 0):

        # Load a random image of actor 0
        warped, img = dataset[0][random.randint(0,len(dataset[0])-1)]
             
        # Use decoder of actor 1
        model.setMode(1)

        output = model(torch.from_numpy(np.asarray([img])).float().cuda())
        output = to_img(output.cpu().data.numpy()[0]*255)
        
        cv2.imwrite("./dc_img/Input_nihesh"+str(epoch)+".jpg", to_img(img*255))
        cv2.imwrite("./dc_img/Input_nihesh_warped"+str(epoch)+".jpg", to_img(warped*255))
        cv2.imwrite("./dc_img/Output_harsh"+str(epoch)+".jpg", output)     
        
        # Use decoder of actor 0
        model.setMode(0)

        output = model(torch.from_numpy(np.asarray([img])).float().cuda())
        output = to_img(output.cpu().data.numpy()[0]*255)
        
        cv2.imwrite("./dc_img/Output_nihesh"+str(epoch)+".jpg", output)   
    
    if((epoch+1)%MODEL_SAVE_RATE == 0):
        torch.save(model.state_dict(), './model/conv_autoencoder-loss-'+str(round(sum(cum_loss)/tot,6))+'.pth')
    
    print()
