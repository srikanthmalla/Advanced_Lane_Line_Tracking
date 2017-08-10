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
		self.combined=False
	def save_image(self,image_name,output):
		#save output image
		# output=write_text(output,self.height,self.width,self.curverad,self.offset,self.combined)
		output_path='./output_images/'+image_name
		scipy.misc.imsave(output_path, output)
	def process_image(self,image1):
		output = np.zeros((self.height,self.width,3),dtype=np.uint8)

		#undistorted images
		undist_img=cc.undistort(image1,self.mtx,self.dist)
		# self.save_image('2_undistorted.jpg',undist_img)

		#thresholding
		binary=transform.threshold(undist_img)
		# self.save_image('3_binary.jpg',binary)

		#warping
		img_size=(binary.shape[1],binary.shape[0])
		binary_warped=transform.warp(binary,img_size)
		# self.save_image('4_binary_warped.jpg',binary_warped)

		img_size=(undist_img.shape[1],undist_img.shape[0])
		color_warped=transform.warp(undist_img,img_size)
		# self.save_image('5_color_warped.jpg',color_warped)

		#Detect Lines
		points,self.curverad,self.offset=detect_lanes(binary_warped[:,:,1])
		print(self.offset)
		lane=draw(color_warped,points)
		# self.save_image('6_lane.jpg',lane)

		#Final output with detected lanes
		final=transform.unwarp(lane,img_size)
		dst=cv2.addWeighted( undist_img, 1, final, 0.3, 0.0);
		# self.save_image('7_final.jpg',dst)

		if self.combined:
			output[0:int(self.height/2),0:int(self.width/3)]=scipy.misc.imresize(image1,(int(self.height/2),int(self.width/3)))
			output[0:int(self.height/2),int(self.width/3):2*int(self.width/3)]=scipy.misc.imresize(undist_img,(int(self.height/2),int(self.width/3)))
			output[0:int(self.height/2),2*int(self.width/3):self.width]=scipy.misc.imresize(binary,(int(self.height/2),int(self.width/3)))
			output[int(self.height/2):self.height,0:int(self.width/3)]=scipy.misc.imresize(binary_warped,(int(self.height/2),int(self.width/3)))
			output[int(self.height/2):self.height,int(self.width/3):2*int(self.width/3)]=scipy.misc.imresize(lane,(int(self.height/2),int(self.width/3)))
			output[int(self.height/2):self.height,2*int(self.width/3):self.width]=scipy.misc.imresize(dst,(int(self.height/2),int(self.width/3)))
		else:
			output=scipy.misc.imresize(dst,(int(self.height),int(self.width)))

		# output=np.array(output,dtype=np.uint8)
		output=np.uint8(output)
		output=write_text(output,self.height,self.width,self.curverad,self.offset, self.combined)
		return output

	def process_video(self,video_name):
		output_v = 'output_videos/'+video_name
		clip1 = VideoFileClip(video_name)
		print('processing video..')
		clip = clip1.fl_image(self.process_image)
		clip.write_videofile(output_v, audio=False)