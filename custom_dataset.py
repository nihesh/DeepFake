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
import cv2
import random
from skimage.transform import swirl

IMAGE_SIZE = (128, 192) 
WARP_PROB = 0.9

class CustomDataset(Dataset):

    def __init__(self, path, mode):

        self.ROOT = path
        self.ACTORS = ["nihesh", "harsh"]
        
        self.path = self.ROOT + self.ACTORS[mode] + "_train/"

        self.files = os.listdir(self.path)
    

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        global WARP_PROB, IMAGE_SIZE

        file_path = self.path + self.files[idx]
        img = cv2.imread(file_path)
    
        # Resize image to reduce computation complexity
        img = scipy.misc.imresize(img, (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
        warped = swirl(img,rotation=0, strength=random.uniform(0.6,1.5), radius=random.randint(300,600),  mode = "constant")        
        
        img = np.rollaxis(img,2,0)
        img = img.astype(float)

        warped = np.rollaxis(warped, 2, 0)
        warped = warped.astype(float)
               
        # Image normalisation
        img = img/255
        
        warp_prob = random.uniform(0,1)
        if(warp_prob < WARP_PROB):
            return warped, img
        else:
            return img, img
