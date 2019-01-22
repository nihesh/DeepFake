# File: autoencoder.py
# Authors:
# Nihesh Anderson
# Harsh Pathak
# Date : Jan 16, 2019

import torch
import torchvision
from torch import nn
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

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
}
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

BLUR_FILTER_SIZE = 23

def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = np.rollaxis(x, 0, 3)
    return x

class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.blurring = nn.Conv2d(3,3,BLUR_FILTER_SIZE, padding=BLUR_FILTER_SIZE//2, bias=False)
		self.blurring.weight = torch.nn.Parameter((torch.ones(BLUR_FILTER_SIZE,BLUR_FILTER_SIZE)*(1/(BLUR_FILTER_SIZE*BLUR_FILTER_SIZE))).expand(self.blurring.weight.size()), requires_grad=False)
		self.encoder = nn.Sequential(
			self.blurring,                                  # 360 x 240 => 360 x 240 - Blur effect
			nn.Conv2d(3, 64, 3, stride=3, padding=0),       # 360 x 240 => 120 x 80 
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),  					# 120 x 80  => 60 x 40
			nn.Conv2d(64, 128, 2, stride=2, padding=0),  		# 60 x 40 => 30 x 20
			nn.ReLU(True),
			# nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 2, stride=2),  		# 30 x 20 => 60 x 40
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),  # 60 x 40 => 120 x 80
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 3, stride=3, padding=0),  # 120 x 80 => 360 x 240
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

class LinearBlur():
	def __init__(self):
		self.conv = nn.Conv2d(3,3,BLUR_FILTER_SIZE, padding=BLUR_FILTER_SIZE//2, bias=False)
		self.conv.weight = torch.nn.Parameter((torch.ones(BLUR_FILTER_SIZE,BLUR_FILTER_SIZE)*(1/(BLUR_FILTER_SIZE*BLUR_FILTER_SIZE))).expand(self.conv.weight.size()), requires_grad=False)
		self.linear_blur = nn.Sequential(
			self.conv
		)

	def forward(self, x):
		x = self.linear_blur(x)
		return x


model = autoencoder().cuda()
blur = LinearBlur()
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
