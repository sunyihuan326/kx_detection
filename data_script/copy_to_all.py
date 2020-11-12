# -*- coding: utf-8 -*-
# @Time    : 2020/11/10
# @Author  : sunyihuan
# @File    : copy_to_all.py
'''
拷贝文件夹下，所有文件夹中的图片至同一个文件中

'''
import os
import shutil

img_root = "F:/serve_data/202011101703"
for c in os.listdir(img_root):
    for c_img in os.listdir(img_root + "/" + c):
        shutil.move(img_root + "/" + c + "/" + c_img, img_root + "/" + c_img)
