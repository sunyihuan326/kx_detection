# -*- coding: utf-8 -*-
# @Time    : 2020/6/24
# @Author  : sunyihuan
# @File    : cut_txt_some_data.py
'''
txt中删除部分数据
'''
import os
txt_path = "E:/kx_detection/multi_detection/data/dataset/202005_1/test.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

txt_file_new_list=[]
for txt_file_one in txt_files:
    if len(txt_file_one.split(" "))>2:
        txt_file_new_list.append(txt_file_one)

print(len(txt_file_new_list))
new_txt_name = "E:/kx_detection/multi_detection/data/dataset/202005_1/test__.txt"
file = open(new_txt_name, "w")
for i in txt_file_new_list:
    file.write(i)


