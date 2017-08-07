import scripts.camera_calibration as cc
import scripts.transform as transform
from scripts.helper_funcs import write_text,detect_lanes,draw
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

class process:
	def __init__(self):
		#calibration
		calib_images_dir='./camera_cal/'
		self.height=1000
		self.width=1500
		self.mtx,self.dist=cc.calibrate(calib_images_dir)
	def save_image(self,image_name,output):
		#save output image
		output_path='./output_images/'+image_name
		scipy.misc.imsave(output_path, output)
		write_text(output_path,self.height,self.width)
	def process_image(self,image1):
		output = np.zeros((self.height,self.width,3))
		#distorted image	
		output[0:int(self.height/2),0:int(self.width/3)]=scipy.misc.imresize(image1,(int(self.height/2),int(self.width/3)))

		#undistorted images
		undist_img=cc.undistort(image1,self.mtx,self.dist)
		output[0:int(self.height/2),int(self.width/3):2*int(self.width/3)]=scipy.misc.imresize(undist_img,(int(self.height/2),int(self.width/3)))

		#thresholding
		binary=transform.threshold(undist_img)
		output[0:int(self.height/2),2*int(self.width/3):self.width]=scipy.misc.imresize(binary,(int(self.height/2),int(self.width/3)))

		#warping
		img_size=(binary.shape[1],image1.shape[0])
		binary_warped=transform.warp(binary,img_size)

		img_size=(undist_img.shape[1],undist_img.shape[0])
		color_warped=transform.warp(undist_img,img_size)

		output[int(self.height/2):self.height,0:int(self.width/3)]=scipy.misc.imresize(binary_warped,(int(self.height/2),int(self.width/3)))

		#Detect Lines
		points=detect_lanes(binary_warped[:,:,1])
		lane=draw(color_warped,points)
		output[int(self.height/2):self.height,int(self.width/3):2*int(self.width/3)]=scipy.misc.imresize(lane,(int(self.height/2),int(self.width/3)))

		#Final output with detected lanes
		final=transform.unwarp(lane,img_size)
		dst=cv2.addWeighted( undist_img, 0.9, final, 0.4, 0.0);

		output[int(self.height/2):self.height,2*int(self.width/3):self.width]=scipy.misc.imresize(dst,(int(self.height/2),int(self.width/3)))

		return output

	def process_video(self,video_name):
		output_v = 'output_videos/'+video_name
		clip1 = VideoFileClip(video_name)
		print('processing video..')
		clip = clip1.fl_image(self.process_image)
		clip.write_videofile(output_v, audio=False)