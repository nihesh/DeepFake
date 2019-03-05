import cv2
from scipy import ndimage
import os
import random

src  = ["./data/nihesh", "./data/harsh"]
for i in src:
    
    if(not os.path.exists(i+"_train")):
        os.mkdir(i+"_train")

    vidcap = cv2.VideoCapture( i + '.mp4')
    success,image = vidcap.read()
    
    count = 0  
    while success:
                    
        cv2.imwrite(i + "_train/hframe%d.jpg" % count, image)     # save frame as JPEG file      
        count+=1
        success,image = vidcap.read()

    print(i+" processed")

