# -*- encoding: utf-8 -*-

"""
从txt文件种拷贝复制部分数据，如：RoastedChicken类等

@File    : copy_data_from_txt.py
@Time    : 2019/12/24 18:53
@Author  : sunyihuan
"""

txt_path = "/multi_detection/data/dataset/XandOld/serve_train0926_oldAndX1217_cut_pizzatwo.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_0_list = []
for txt_file_one in txt_files:
    if "RoastedChicken.jpg 0" in txt_file_one:
        train_0_list.append(txt_file_one)

for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
for i in range(len(train_0_list)):
    txt_files.append(train_0_list[i])
print(len(txt_files))

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/XandOld/serve_train0926_oldAndX1217_cut_pizzatwo_roasted.txt"
file = open(new_txt_name, "w")
for i in txt_files:
    file.write(i)
