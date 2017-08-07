from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def write_text(img_path,height,width):
	img = Image.open(img_path)
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype("arial.ttf", 20)
	# draw.text((x, y),"Sample Text",(r,g,b))
	draw.text((20,0),"Distorted Image",(255,255,255),font=font)
	draw.text((width/2+20,0),"Undistorted Image",(255,255,255),font=font)
	draw.text((20,height/2),"Gradient Thresholded Image",(255,255,255),font=font)

	img.save(img_path)