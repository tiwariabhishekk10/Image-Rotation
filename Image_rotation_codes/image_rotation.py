import cv2
import os
import numpy as np
import imutils

from numpy.random import randint

outpath=r'D:\Python Code\Image_rotation\rotated_image'
path=r'D:\Python Code\Image_rotation\image'

f = open("new_image_rotation.txt", 'a')

for image_path in os.listdir(path):
    input_path=os.path.join(path,image_path)
    image_to_rotate=cv2.imread(input_path)
    angle=randint(1,359)
    rotated=imutils.rotate_bound(image_to_rotate,angle)#differnce between rotate and rotate_bound is i does not cuts any part of image.

    fullpath=os.path.join(outpath,'rotated_'+image_path)
    f.write(image_path+" " +str(angle)+"\n")
    cv2.imwrite(fullpath,rotated)
f.close()
