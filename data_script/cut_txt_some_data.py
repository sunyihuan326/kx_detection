# -*- coding: utf-8 -*-
# @Time    : 2020/6/24
# @Author  : sunyihuan
# @File    : cut_txt_some_data.py
'''
txt中删除部分数据
'''
import os


def cut_txt_data(src, dst, target_str):
    txt_file = open(src, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    # if target_str=="len2":
    #     for txt_file_one in txt_files:
    #         if len(txt_file_one.split(" ")) > 2:
    #             txt_file_new_list.append(txt_file_one)
    txt_file_new_list = []
    for txt_file_one in txt_files:
        if target_str[0] in txt_file_one and target_str[1] in txt_file_one:
            pass
        else:
            txt_file_new_list.append(txt_file_one)

    print(len(txt_file_new_list))

    file = open(dst, "w")
    for i in txt_file_new_list:
        file.write(i)


txt_path = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand_900hotdog.txt"
new_txt_name = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand_900hotdog__.txt"

cut_txt_data(txt_path, new_txt_name,["20210115",""])
