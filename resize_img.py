# -*- coding: utf-8 -*-

'''
将图片填充按最长边成为正方形,再resize成416*416,并保存下来
'''

from PIL import Image
import os

img_names=os.listdir('data')
# print(img_names)
for i in range(len(img_names)):
    name=img_names[i]
    print(name)
    img=Image.open(os.path.join('data',name))
    #PIL打开图片的形状是whc(注：c没有显示，OpenCV打开是hwc，c是显示的)
    w,h=img.size
    # print(w,h)
    #最大边
    max_size=max(w,h)
    #以最长边为边长创建新的正方形图片，背景为白色(255,255,255)，黑色是(0,0,0)
    new_img=Image.new('RGB',(max_size,max_size),color=(255,255,255))
    #将img由在中间粘贴
    new_img.paste(img,box=(int((max_size-w)/2), int((max_size-h)/2)))
    # new_img.show()
    new_img=new_img.resize((416,416),Image.ANTIALIAS)
    new_img.save(f'data1/images/{i}.jpg')


