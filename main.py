from scripts.process import *
p=process()

#image processing pipeline
# from os import listdir
# from os.path import isfile, join
# mypath='./test_images'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# for image_name in onlyfiles: 	
# 	print('\n')
# 	print('--'*10)
# 	print(image_name)
# 	print('--'*10)
# 	test_images_dir='./test_images/'
# 	#distorted images
# 	image1=mpimg.imread(test_images_dir+image_name)
# 	output=p.process_image(image1)
# 	p.save_image(image_name,output)

#video processing pipeline
video_name='harder_challenge_video.mp4'
p.process_video(video_name)