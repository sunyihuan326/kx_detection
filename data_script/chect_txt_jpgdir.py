# -*- coding: utf-8 -*-
# @Time    : 2020/5/19
# @Author  : sunyihuan
# @File    : chect_txt_jpgdir.py

import os

jpg_path = "E:/DataSets/X_data_27classes/JPGImages"
train_txt = "E:/DataSets/X_data_27classes/train.txt"
test_txt = "E:/DataSets/X_data_27classes/test.txt"
val_txt = "E:/DataSets/X_data_27classes/val.txt"
all_jpg_list = [b.split(".jpg")[0] for b in os.listdir(jpg_path)]
print(len(all_jpg_list))

train_txt_file = open(train_txt, "r")
train_txt_files = train_txt_file.readlines()
print(len(train_txt_files))
print(len(list(set(train_txt_files))))
test_txt_file = open(test_txt, "r")
test_txt_files = test_txt_file.readlines()
print(len(list(set(test_txt_files))))
val_txt_file = open(val_txt, "r")
val_txt_files = val_txt_file.readlines()
print(len(val_txt_files))
print(len(list(set(val_txt_files))))
txt_list=list(set(train_txt_files)|set(test_txt_files)|set(val_txt_files))
print(len(txt_list))


