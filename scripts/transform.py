import cv2
import numpy as np
import matplotlib.pyplot as plt

# src = np.float32(
# [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
# [((img_size[0] / 6) ), img_size[1]],
# [(img_size[0] * 5 / 6) , img_size[1]],
# [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])

# dst = np.float32(
# [[(img_size[0] / 4), 0],
# [(img_size[0] / 4), img_size[1]],
# [(img_size[0] * 3 / 4), img_size[1]],
# [(img_size[0] * 3 / 4), 0]])
def warp(img,img_size):
	img_size=(img.shape[1],img.shape[0])
	src = np.float32(
	[[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
	[((img_size[0] / 5) ), img_size[1]],
	[(img_size[0] * 4 / 5) , img_size[1]],
	[(img_size[0] / 2 + 50), img_size[1] / 2 + 100]])

	dst = np.float32(
	[[(img_size[0] / 4), 0],
	[(img_size[0] / 4), img_size[1]],
	[(img_size[0] * 3 / 4), img_size[1]],
	[(img_size[0] * 3 / 4), 0]])

	M=cv2.getPerspectiveTransform(src,dst)
	warped=cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
	# warped=np.expand_dims(warped,axis=2)
	# warped=np.tile(warped,(1,1,3))
	# plt.imshow(warped)
	# plt.show()
	return warped
def unwarp(img,img_size): #perspective transform	
	img_size=(img.shape[1],img.shape[0])
	src = np.float32(
	[[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
	[((img_size[0] / 5) ), img_size[1]],
	[(img_size[0] * 4 / 5) , img_size[1]],
	[(img_size[0] / 2 + 50), img_size[1] / 2 + 100]])

	dst = np.float32(
	[[(img_size[0] / 4), 0],
	[(img_size[0] / 4), img_size[1]],
	[(img_size[0] * 3 / 4), img_size[1]],
	[(img_size[0] * 3 / 4), 0]])

	M=cv2.getPerspectiveTransform(dst,src)
	unwarped=cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
	return unwarped
def threshold(image, s_thresh=(150, 255), sx_thresh=(20, 100)):
	image = np.copy(image)
	# https://www.packtpub.com/mapt/book/Application+Development/9781785283932/2/ch02lvl1sec22/Sharpening
	kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
								 [-1,2,2,2,-1],
								 [-1,2,8,2,-1],
								 [-1,2,2,2,-1],
								 [-1,-1,-1,-1,-1]]) / 8.0

	image = cv2.filter2D(image, -1, kernel_sharpen_3)

	hls = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
#     gray = cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_RGB2GRAY)
	gray = (0.5*image[:,:,0] + 0.4*image[:,:,1] + 0.1*image[:,:,2]).astype(np.uint8)
	s = hls[:,:,2]
	l = hls[:,:,1]

	_, gray_binary = cv2.threshold(gray.astype('uint8'), 150, 255, cv2.THRESH_BINARY)

	total_px = image.shape[0]*image.shape[1]
	laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
	mask_three = (laplacian < 0.15*np.min(laplacian)).astype(np.uint8)
	if cv2.countNonZero(mask_three)/total_px < 0.01:
		laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
		mask_three = (laplacian < 0.075*np.min(laplacian)).astype(np.uint8)

	_, s_binary = cv2.threshold(s.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
	_, l_binary = cv2.threshold(l.astype('uint8'), 200, 255, cv2.THRESH_BINARY)
	mask_two = s_binary

	combined_binary = np.clip(cv2.bitwise_and(gray_binary, 
						cv2.bitwise_or(mask_three,cv2.bitwise_or(l_binary, mask_two))), 0, 255).astype('uint8')
	# combined_binary = np.clip(l_binary, 0, 255).astype('uint8')

	combined_binary=np.expand_dims(combined_binary,axis=2)
	combined_binary=np.tile(combined_binary,(1,1,3))
	# print(np.shape(combined_binary))
	# plt.imshow(combined_binary)
	# plt.show()
	return combined_binary

