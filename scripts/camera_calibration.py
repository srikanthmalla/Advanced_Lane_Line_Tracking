import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

input_dir='../camera_cal/'
def chessboardcorners(image):
	#Find CHess board corners
	ret,corners=cv2.findChessboardCorners(gray,(8,6),None)
	if ret == True:
		imgPoints.append(corners)
		objPoints.append(objP)
		#draw and display corners
		img=cv2.drawChessboardCorners(img,(8,6),corners,ret)
		plt.show(img)
		return (objPoints,imgPoints)
def calibrate(input_dir):
	images=glob.glob(input_dir+'calibration*.jpg')

	objPoints=[]#3d points
	imgPoints=[]#2d points
	#prepare object points like (0,0,0),(1,0,0),(2,0,0),....(7,5,0)
	objP=np.zeros((6*8,3),np.float32)
	objP[:,:2]=mgrid[0:8,0:6].T.reshape(-1,2)#x,y coordinates

	for fname in images:
		img=mpimg.imread(fname)
		#convert image to gray
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		objPoints,imgPoints=chessboardcorners(gray,objPoints,imgPoints)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
	return (mtx,dist)
def undistort(img,mtx,dist):
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist