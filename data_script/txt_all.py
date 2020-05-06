# -*- encoding: utf-8 -*-

"""
@File    : txt_all.py
@Time    : 2019/12/6 15:39
@Author  : sunyihuan
"""

import random

train_all_list = []
txt_path = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/1111_train.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

for txt_file_one in txt_files:
    if txt_file_one not in train_all_list:
        train_all_list.append(txt_file_one)
    else:
        continue

txt_path_1203 = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/1203_train.txt"
txt_file_1203 = open(txt_path_1203, "r")
txt_files_1203 = txt_file_1203.readlines()
print(len(txt_files_1203))

for txt_file_one in txt_files_1203:
    if txt_file_one not in train_all_list:
        train_all_list.append(txt_file_one)
    else:
        continue

txt_path_1205 = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/1205_train.txt"
txt_file_1205 = open(txt_path_1205, "r")
txt_files_1205 = txt_file_1205.readlines()
print(len(txt_files_1205))

for txt_file_one in txt_files_1203:
    if txt_file_one not in train_all_list:
        train_all_list.append(txt_file_one)
    else:
        continue

print(len(train_all_list))

new_txt_name = "E:/DataSets/KX_FOODSets_model_data/20191206data/ImageSets/Main/train_all.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
