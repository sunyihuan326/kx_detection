# -*- encoding: utf-8 -*-

"""
@File    : peanuts_txt_change.py
@Time    : 2019/11/28 14:08
@Author  : sunyihuan
"""
txt_path = "E:/kx_detection/multi_detection/data/dataset/peanuts/peanuts_train.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
print(len(txt_files))

train_all_list = []
for txt_file_one in txt_files:
    img_path_name = txt_file_one.split(" ")[0]
    # root_path = img_path_name.split("peanuts")[0]
    print(img_path_name.split("peanuts")[1])
    txt_file_name = "/home/sunyihuan/sunyihuan_algorithm/data/X_KX_data_27_1111_train_aug" + "/peanuts"
    txt_file_name += img_path_name.split("peanuts")[1] + "\n"
    train_all_list.append(txt_file_name)  # 读取一个插入一个

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/peanuts/peanuts_train_new.txt"
file = open(new_txt_name, "w")
for i in train_all_list:
    file.write(i)
