# -*- encoding: utf-8 -*-

"""
@File    : from_img_dir_write_txt.py
@Time    : 2019/12/9 14:29
@Author  : sunyihuan
"""

import os

img_dir = "E:/DataSets/KX_FOODSets_model_data/20191217_X3camera5/JPGImages"

txt_list = []
for img_name in os.listdir(img_dir):
    if img_name.endswith(".jpg"):
        txt_list.append(img_name.split(".jpg")[0] + "\n")

print(len(txt_list))
txt_name = "E:/DataSets/KX_FOODSets_model_data/20191217_X3camera5/ImageSets/Main/train.txt"
file = open(txt_name, "w")
for i in txt_list:
    file.write(i)
