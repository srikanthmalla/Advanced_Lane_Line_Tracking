import scripts.camera_calibration as cc
import scripts.transform as transform
from scripts.helper_funcs import write_text
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

height=1000
width=1000
output = np.zeros((height,width,3))

#calibration
calib_images_dir='./camera_cal/'
mtx,dist=cc.calibrate(calib_images_dir)
test_images_dir='./test_images/'
image_name='test1.jpg'
#distorted images
image1=img=mpimg.imread(test_images_dir+image_name)
output[0:height/2,0:width/2]=scipy.misc.imresize(image1,(height/2,width/2))
#undistorted images
undist_img=cc.undistort(image1,mtx,dist)
output[0:height/2,width/2:width]=scipy.misc.imresize(undist_img,(height/2,width/2))
#thresholding
binary=transform.threshold(undist_img)
output[height/2:height,0:width/2]=scipy.misc.imresize(binary,(height/2,width/2))

#save output image
output_path='./output_images/'+image_name
scipy.misc.imsave(output_path, output)
write_text(output_path,height,width)