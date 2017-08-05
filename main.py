import scripts.camera_calibration as cc
import scripts.transform as transform

input_dir='./camera_cal/'
mtx,dist=cc.calibrate(input_dir)
undist_img=cc.undistort(image)