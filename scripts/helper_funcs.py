from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def write_text(img,height,width,curvature,offset, combined):
	# img = Image.open(img_path)
	img=Image.fromarray(img) 
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype("arial.ttf", 30)
	if combined:
		draw.text((20,0),"Distorted Image",(255,255,255),font=font)
		draw.text((width/3+20,0),"Undistorted Image",(255,255,255),font=font)
		draw.text((2*width/3+20,0),"Gradient Thresholded Image",(255,255,255),font=font)
		
		draw.text((20,height/2),"Warped image",(255,255,255),font=font)
		draw.text((width/3+20,height/2),"Lane detected",(255,255,255),font=font)
		draw.text((2*width/3+20,height/2),"Final Output (w detected lane)",(255,255,255),font=font)

		draw.text((width/2, 100),"curvature (m): "+str(curvature),(255,255,255),font=font)
		draw.text((width/2, 200),"offset from center (m): "+str(offset),(255,255,255),font=font)

	else:
		curve="curvature (m): "+str(curvature)
		draw.text((width/2, 100),curve,(255,255,255),font=font)
		draw.text((width/2, 200),"offset from center (m): "+str(offset),(255,255,255),font=font)
	img=np.asarray(img) 
	return img
	# img.save(img_path)

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

def find_window_centroids(warped, window_width, window_height, margin):
	
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions
	
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template 
	
	# Sum quarter bottom of image to get slice, could use a different ratio
	l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
	l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
	r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
	r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
	
	# Add what we found for the first layer
	window_centroids.append((l_center,r_center))
	
	# Go through each layer looking for max pixel locations
	for level in range(1,(int)(warped.shape[0]/window_height)):
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
		l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
		r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
		# Add what we found for that layer
		window_centroids.append((l_center,r_center))

	return window_centroids
def draw(img,pts):
	cv2.fillPoly(img, np.int_(pts), (0,255, 0))
	return img

def detect_lanes(warped):
	print('detecting lane and fitting polynomial..')
	window_width = 100 
	window_height = 180 # Break image into 9 vertical layers since image height is 720
	margin = 50 # How much to slide left and right for searching
	window_centroids = find_window_centroids(warped, window_width, window_height, margin)
	# If we found any window centers
	if len(window_centroids) > 0:
		# print(window_centroids)
		window_centroids=np.asarray(window_centroids)

		leftx=window_centroids[:,0]
		rightx=window_centroids[:,1]
		lefty=range(warped.shape[0],0,-window_height)
		righty=range(warped.shape[0],0,-window_height)
		left_fit = np.polyfit(lefty,leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		# Points used to draw all the left and right windows
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))
		
		warped=np.expand_dims(warped,axis=2)
		warped=np.tile(warped,(1,1,3))

		# Define conversions in x and y from pixels space to meters
		y_eval = np.max(ploty)
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meteres per pixel in x dimension
		fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
		curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) \
							 /np.absolute(2*fit_cr[0])
		print("curvature:",curverad)
		# Now our radius of curvature is in meters
		# print(left_curverad, 'm', right_curverad, 'm')
		image_shape=np.shape(warped)
		position = image_shape[1]/2
		center = (left_fitx[-1] + right_fitx[-1])/2
		# Define conversions in x and y from pixels space to meters
		xm_per_pix = 3.7/700 # meteres per pixel in x dimension    
		offset= (position - center)*xm_per_pix
		# plt.imshow(warped)
		# plt.plot(left_fitx, ploty, color='yellow')
		# plt.plot(right_fitx, ploty, color='yellow')
		# plt.xlim(0, 1280)
		# plt.ylim(720, 0)
		# plt.show()
		return pts,curverad,offset