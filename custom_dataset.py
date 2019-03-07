# File: autoencoder.py
# Authors:
# Nihesh Anderson
# Date : March 5, 2019

from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.misc

IMAGE_SIZE = (256, 384) 

class CustomDataset(Dataset):

    def __init__(self, path, mode):

        self.ROOT = path
        self.ACTORS = ["nihesh", "harsh"]
        
        self.path = self.ROOT + self.ACTORS[mode] + "_train/"

        self.files = os.listdir(self.path)

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file_path = self.path + self.files[idx]
        img = cv2.imread(file_path)
    
        # Resize image to reduce computation complexity
        img = scipy.misc.imresize(img, (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
        
        img = np.rollaxis(img,2,0)
        img = img.astype(float)
        
        # Image normalisation
        img = img/255

        return img
