# -*- encoding: utf-8 -*-

"""
@File    : cut_txt.py
@Time    : 2019/12/6 11:14
@Author  : sunyihuan
"""

import random

txt_path = "E:/kx_detection/multi_detection/data/dataset/XandOld/train0926_oldAndX1217.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_all_list = []
pizzatwo_list = []
for txt_file_one in txt_files:
    # if "Potato" in txt_file_one:
    #     continue
    if "Pizzatwo" in txt_file_one:
        pizzatwo_list.append(txt_file_one)
    # elif "Toast" in txt_file_one:
    #     continue
    else:
        train_all_list.append(txt_file_one)

print("all train:", len(train_all_list))
print("all pizzatwo:", len(pizzatwo_list))

random.shuffle(pizzatwo_list)

for i in range(700):
    train_all_list.append(pizzatwo_list[i])
print("all new train:", len(train_all_list))

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/XandOld/train0926_oldAndX1217_cut_pizzatwo.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
