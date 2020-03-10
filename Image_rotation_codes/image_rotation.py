import cv2
import os
import numpy as np

from numpy.random import randint
from scipy import ndimage,misc#downgraded to 1.2.2

outpath=r'D:\Python Code\Image_rotation\new_rotated_image'
path=r'D:\Python Code\Image_rotation\image'

f = open("new_image_rotation2.txt", 'a')

for image_path in os.listdir(path):
    input_path=os.path.join(path,image_path)
    image_to_rotate=ndimage.imread(input_path)
    angle=randint(1,359)
    rotated=ndimage.rotate(image_to_rotate,angle)

    fullpath=os.path.join(outpath,'rotated_'+image_path)
    f.write(image_path+" " +str(angle)+"\n")
    misc.imsave(fullpath,rotated)
f.close()
