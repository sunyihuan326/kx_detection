# -*- encoding: utf-8 -*-

"""
将文件夹中的所有图片名写入到txt文件中

@File    : write_txt_from_img_dir.py
@Time    : 2019/12/9 14:29
@Author  : sunyihuan
"""

import os

img_dir = "E:/DataSets/X_3660_data/nofood/JPGImages"

txt_list = []
for img_name in os.listdir(img_dir):
    if img_name.endswith(".jpg"):
        txt_list.append(img_name.split(".jpg")[0] + "\n")

print(len(txt_list))
txt_name = "E:/DataSets/X_3660_data/nofood/ImageSets/Main/train.txt"
file = open(txt_name, "w")
for i in txt_list:
    file.write(i)
