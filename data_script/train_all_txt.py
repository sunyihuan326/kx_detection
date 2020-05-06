#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 
# @Author  : sunyihuan

'''
将各类别的txt文件统一写入到一个txt文件中

如：beefsteak_train.txt  chickenwings_train.txt都写到train.txt中
'''

import os, shutil
import random

root_path = "C:/Users/sunyihuan/Desktop/20191205/ImageSets/Main"
train_all_list = []


def train_all_txt(txt_name):
    '''
    读取所有的行
    :param txt_name: txt文件名称
    :return:
    '''
    txt_name = root_path + "/" + txt_name
    txt_file = open(txt_name, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))
    for txt_file_one in txt_files:
        train_all_list.append(txt_file_one)  # 读取一个插入一个
    return train_all_list


if __name__ == "__main__":
    all_txt_name = "test_all.txt"  # 写入到train文件中
    # all_txt_name = "test_all.txt"  # 写入到test文件中
    for txt_name in os.listdir(root_path):
        if "_test" in txt_name:
            train_all_list = train_all_txt(txt_name)
    random.shuffle(train_all_list)
    print(len(train_all_list))
    all_txt_name = root_path + "/" + all_txt_name
    file = open(all_txt_name, "w")
    for i in train_all_list:
        file.write(i)
