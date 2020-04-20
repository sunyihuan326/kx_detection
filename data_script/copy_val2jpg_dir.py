# -*- coding: utf-8 -*-
# @Time    : 2020/4/7
# @Author  : sunyihuan
# @File    : copy_val2jpg_dir.py

'''
将xxx_val.txt图片拷贝到对应的xxx_val文件中
'''
import os
import shutil


def copy_val2jpg_dir(txt_root, all_jpg_dir, save_dir):
    for txt in os.listdir(txt_root):
        if "_val" in txt:
            os.mkdir(save_dir + "/" + txt.split(".")[0])
            val_jpg_names = open(txt_root + "/" + txt).readlines()
            for jpg_name in val_jpg_names:
                jpg_name=jpg_name.strip("\n")
                jpg_name = jpg_name + ".jpg"
                shutil.copy(all_jpg_dir+ "/" + jpg_name, save_dir + "/" + txt.split(".")[0] + "/" + jpg_name)


txt_root = "E:/DataSets/2020_two_phase_KXData/only2phase_data/20200402/ImageSets/Main"
all_jpg_dir = "E:/DataSets/2020_two_phase_KXData/only2phase_data/20200402/JPGImages"
save_dir = "E:/DataSets/2020_two_phase_KXData/only2phase_data/20200402/JPGImages_val"
copy_val2jpg_dir(txt_root, all_jpg_dir, save_dir)
