# -*- encoding: utf-8 -*-

"""
@File    : from_img_dir_write_txt.py
@Time    : 2019/12/9 14:29
@Author  : sunyihuan
"""

import os

img_dir = "E:/DataSets/2020_two_phase_KXData/all_data36classes/JPGImages/test_resize"

txt_list = []
for img_name in os.listdir(img_dir):
    if img_name.endswith(".jpg"):
        txt_list.append(img_name.split(".jpg")[0] + "\n")

print(len(txt_list))
txt_name = "E:/DataSets/2020_two_phase_KXData/all_data36classes/JPGImages/test_resize.txt"
file = open(txt_name, "w")
for i in txt_list:
    file.write(i)
