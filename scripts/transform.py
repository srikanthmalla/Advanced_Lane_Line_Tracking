import cv2
import numpy as np
import matplotlib.pyplot as plt
def warp(img,src,dst):
	img_size=(img.shape[1],img.shape[0])
	M=cv2.getPerspectiveTransform(src,dst)
	warped=cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
	return warped
def unwarp(img,src,dst): #perspective transform
	img_size=(img.shape[1],img.shape[0])
	M=cv2.getPerspectiveTransform(dst,src)
	unwarped=cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
	return unwarped
def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
    combined_binary=np.expand_dims(combined_binary,axis=2)
    combined_binary=np.tile(combined_binary,(1,1,3))
    # print(np.shape(combined_binary))
    # plt.imshow(combined_binary)
    # plt.show()
    return combined_binary