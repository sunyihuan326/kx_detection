# -*- coding: utf-8 -*-
# @Time    : 2020/6/3
# @Author  : sunyihuan
# @File    : new_jpg.py
'''
随意生成一张图片

'''
from PIL import Image

Image.new("RGB", (800, 600), (255, 255, 255)).save("86.jpg", "JPG")
