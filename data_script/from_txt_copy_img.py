# -*- encoding: utf-8 -*-

"""
从txt文件中读取图片地址，并将图片保存至统一文件夹

@File    : from_txt_copy_img.py
@Time    : 2019/12/5 16:52
@Author  : sunyihuan
"""
import shutil

txt_path = "E:/kx_detection/multi_detection/data/dataset/XandOld/test0926_oldAndX1206.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

for file in txt_files:
    img_name = file.split(" ")[0]
    jpg_name = str(img_name).split("/")[-1]
    shutil.copy(img_name, "E:/DataSets/KX_FOODSets_model_data/XandOld1206/test/" + jpg_name)
