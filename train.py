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

def to_img(x):
    
    x = np.rollaxis(x, 0, 3)
    return x

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

if not os.path.exists('./dc_img/train'):
	os.mkdir('./dc_img/train')

if not os.path.exists('./dc_img/test'):
	os.mkdir('./dc_img/test')

src  = [ "./data/harsh_train",  "./data/harsh_test"]


num_epochs = 400
batch_size = 64
learning_rate = 1e-2
NORM_VAL = 255

data_dir = "./data"

image_datasets = {}

for x in ["train","test"]:
	
	img = []
	folder = data_dir+"/harsh_"+x+"/1"
	for file in os.listdir(folder):
		my_img = cv2.imread(folder+"/"+file)
		my_img = scipy.misc.imresize(my_img, (240,360,3))
		my_img = np.rollaxis(my_img,2,0)
		my_img = my_img.astype(float)
		img.append(my_img/NORM_VAL)
	img = np.asarray(img)
	image_datasets[x] = img

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers = 12)

              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

image_datasets = None

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
	for phase in ['train', 'test']:
		cum_loss = []
		tot = 0
		for data in dataloaders[phase]:
			img = data.float()
			img = Variable(img).cuda()
			# ===================forward=====================
			output = model(img)
			loss = criterion(output, img)
			cum_loss.append(loss.item()*(data.size()[0]))
			tot+=data.size()[0]
			# ===================backward====================
			if(phase == 'train'):
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# ===================log=========================
		
		if(phase == "train"):
			print(phase.upper()+" pixels mean: "+str(np.mean(img.cpu().data.numpy()))+" pixels max "+str(np.max(img.cpu().data.numpy())))
			print(phase.upper() + ' epoch ['+str(epoch+1)+'/'+str(num_epochs)+'], loss:'+str(sum(cum_loss)/tot))
		
		if epoch % 10 == 0:
			pic = to_img(output.cpu().data.numpy()[0]*NORM_VAL)
			input_pic = to_img(img.cpu().data.numpy()[0]*NORM_VAL)
			# print(np.mean(pic), np.mean(input_pic)) # Input mean - 40, output mean - 0.14 (wait for it to learn and reduce input dim)
			cv2.imwrite('./dc_img/' + phase + '/image_{}.jpg'.format(epoch), pic)
			cv2.imwrite('./dc_img/' + phase + '/image_{}_input.jpg'.format(epoch), input_pic)


torch.save(model.state_dict(), './conv_autoencoder.pth')
