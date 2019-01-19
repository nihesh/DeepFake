import cv2
from scipy import ndimage
import os
import random

src  = ["./data/nihesh", "./data/harsh"]
for i in src:
	if(not os.path.exists(i+"_train")):
		os.mkdir(i+"_train")
		os.mkdir(i+"_test")
	vidcap = cv2.VideoCapture( i + '.mp4')
	success,image = vidcap.read()
	count = 0
	while success:
		# image = ndimage.rotate(image, 90)
		var = random.randint(1,10)
		if(var<=9):
			cv2.imwrite(i + "_train/hframe%d.jpg" % count, image)     # save frame as JPEG file      
		else:
			cv2.imwrite(i + "_test/hframe%d.jpg" % count, image)     # save frame as JPEG file
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1