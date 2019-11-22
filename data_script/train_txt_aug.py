# -*- encoding: utf-8 -*-

"""
@File    : train_txt_aug.py
@Time    : 2019/11/21 14:28
@Author  : sunyihuan
"""

train_txt = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111/ImageSets/Main/train.txt"
txt_file = open(train_txt, "r")
txt_files = txt_file.readlines()
train_all_list = []
for txt_file_one in txt_files:
    txt_new_one = txt_file_one.strip() + "_bright9"
    train_all_list.append(txt_new_one)

txt_new_files = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111_train_aug/bright9/ImageSets/Main/train.txt"
file = open(txt_new_files, "w")
for i in train_all_list:
    file.write(i + "\n")
