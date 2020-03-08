import cv2
import os
import numpy as np

from numpy.random import randint
from scipy import ndimage,misc#downgraded to 1.2.2

outpath='D:\Python Code'
path='D:\Python Code\Images'

f = open("Image_rotation2.txt", 'a')

for image_path in os.listdir(path):
    input_path=os.path.join(path,image_path)
    image_to_rotate=ndimage.imread(input_path)

    for angle in randint(1,359,360):
        rotated=ndimage.rotate(image_to_rotate,angle)

    fullpath=os.path.join(outpath,'rotated_'+image_path)
    f.write(image_path+" " +str(angle)+"\n")
    misc.imsave(fullpath,rotated)
f.close()