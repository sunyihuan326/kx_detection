# -*- encoding: utf-8 -*-

"""
直接更改txt中图片地址
@File    : txt_aug.py
@Time    : 2019/12/6 19:50
@Author  : sunyihuan
"""

txt_path = "E:/kx_detection/multi_detection/data/dataset/XandOld/test0926_oldAndX1206.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_all_list = []
for txt_file_one in txt_files:
    img_path_name = txt_file_one
    # root_path = img_path_name.split("peanuts")[0]
    print(img_path_name.split("JPGImages")[1])
    txt_file_name = "E:/DataSets/KX_FOODSets_model_data/XandOld1206/JPGImages_hot"
    txt_file_name += img_path_name.split("JPGImages")[1]
    train_all_list.append(txt_file_name)  # 读取一个插入一个

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/XandOld/test0926_oldAndX1206_hot.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    img_ = i.split(".jpg")[0] + "_hot" + ".jpg" + i.split(".jpg")[1]
    file.write(img_)
