# -*- coding: utf-8 -*-
# @Time    : 2020/6/11
# @Author  : sunyihuan
# @File    : img_resize.py
from PIL import Image
img_path="C:/Users/sunyihuan/Desktop/X5_test/611_test/sweetpotatocut/20200611113758.jpg"
img = Image.open(img_path)
img_new = img.resize((800, 600), Image.ANTIALIAS)  # 图片尺寸变化

img_new.save("C:/Users/sunyihuan/Desktop/X5_test/611_test/sweetpotatocut/20200611113758_0.jpg")