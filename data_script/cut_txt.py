# -*- encoding: utf-8 -*-

"""
@File    : cut_txt.py
@Time    : 2019/12/6 11:14
@Author  : sunyihuan
"""

import random

txt_path = "E:/已标数据备份/20191203数据清洗补充/ImageSets/val.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_all_list = []
for txt_file_one in txt_files:
    # if "Potato" in txt_file_one:
    #     continue
    if "Pizza" in txt_file_one:
        continue
    # elif "Toast" in txt_file_one:
    #     continue
    else:
        train_all_list.append(txt_file_one)

print(len(train_all_list))

new_txt_name = "E:/已标数据备份/20191203数据清洗补充/ImageSets/val_new.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
