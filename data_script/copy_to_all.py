# -*- coding: utf-8 -*-
# @Time    : 2020/11/10
# @Author  : sunyihuan
# @File    : copy_to_all.py
'''
拷贝文件夹下，所有文件夹中的图片至同一个文件中

'''
import os
import shutil
from tqdm import tqdm

img_root = "F:/serve_data/OVEN/202012"
img_save = "F:/serve_data/OVEN/202012/for_test"

for c in tqdm((os.listdir(img_root))):
    if c != "for_test":
        for c_img in os.listdir(img_root + "/" + c):
            for img in os.listdir(img_root + "/" + c + "/covert_jpg"):
                shutil.copy(img_root + "/" + c + "/covert_jpg" + "/" + img, img_save + "/" + img)
