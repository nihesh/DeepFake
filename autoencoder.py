__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
test_data  = MNIST("./testdata", transform=img_transform, download=True, train = False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

BLUR_FILTER_SIZE = 23

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv = nn.Conv2d(1,1,BLUR_FILTER_SIZE, padding=BLUR_FILTER_SIZE//2, bias=False)
        self.conv.weight = torch.nn.Parameter((torch.ones(BLUR_FILTER_SIZE,BLUR_FILTER_SIZE)*(1/(BLUR_FILTER_SIZE*BLUR_FILTER_SIZE))).expand(self.conv.weight.size()), requires_grad=False)
        self.encoder = nn.Sequential(
            self.conv,
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LinearBlur():
    def __init__(self):
        self.conv = nn.Conv2d(1,1,BLUR_FILTER_SIZE, padding=BLUR_FILTER_SIZE//2, bias=False)
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
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch ['+str(epoch+1)+'/'+str(num_epochs)+'], loss:'+str(loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        input_pic = to_img(img.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
        save_image(blur.forward(input_pic), './dc_img/image_{}_input.png'.format(epoch))

    loss_vec = []
    for data in test_data_loader:
        img, _ = data
        img = Variable(img).cuda()
        output = model(img)
        loss = criterion(output, img)
        loss_vec.append(loss.item())
    
    print('Test epoch ['+str(epoch+1)+'/'+str(num_epochs)+'], loss:'+str(np.mean(loss_vec)))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        input_pic = to_img(img.cpu().data)
        save_image(pic, './dc_img_test/image_{}.png'.format(epoch))
        save_image(blur.forward(input_pic), './dc_img_test/image_{}_input.png'.format(epoch))


torch.save(model.state_dict(), './conv_autoencoder.pth')