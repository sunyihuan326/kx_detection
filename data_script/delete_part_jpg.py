# -*- coding: utf-8 -*-
# @Time    : 2020/5/19
# @Author  : sunyihuan
# @File    : delete_part_jpg.py
import shutil
import os

jpg_path = "C:/Users/sunyihuan/Desktop/JPGImages"
for jpg in os.listdir(jpg_path):
    if "duck" in jpg:
        print(jpg)
        os.remove(jpg_path + "/" + jpg)
