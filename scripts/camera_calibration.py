import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

input_dir='../camera_cal/'
nx=9
ny=6
def chessboardcorners(gray,objPoints,imgPoints,fname):
	#prepare object points like (0,0,0),(1,0,0),(2,0,0),....(7,5,0)
	objP=np.zeros((nx*ny,3),np.float32)
	objP[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)#x,y coordinates
	#Find CHess board corners
	ret,corners=cv2.findChessboardCorners(gray,(nx,ny),None)
	if ret == True:
		imgPoints.append(corners)
		objPoints.append(objP)
		#draw and display corners
		# img=cv2.drawChessboardCorners(gray,(8,6),corners,ret)
		# plt.show(img)
	else:
		print('didnot find corners for %s'%fname)
	return (objPoints,imgPoints)
def calibrate(input_dir):
	print('calibrating..')
	print('--'*10)
	images=glob.glob(input_dir+'calibration*.jpg')

	objPoints=[]#3d points
	imgPoints=[]#2d points
	for fname in images:
		img=mpimg.imread(fname)
		#convert image to gray
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		objPoints,imgPoints=chessboardcorners(gray,objPoints,imgPoints,fname)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
	return (mtx,dist)
def undistort(img,mtx,dist):
	print('undistorting..')
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	# plt.imshow(undist)
	# plt.show()
	return undist