# -*- coding: utf-8 -*-
# @Time    : 2021/3/3
# @Author  : sunyihuan
# @File    : print_txt_data_dir.py
'''
输出txt文件中，数据文件夹
'''

import os


def dir_name(txt_file):
    txt_lists = open(txt_file, "r").readlines()
    names = []
    for tt in txt_lists:
        jpg = tt.split(".jpg")[0]
        dir_names = jpg.split("/")
        dir_name = ""
        for dd in range(len(dir_names) - 1):
            dir_name = os.path.join(dir_name, dir_names[dd])
        if dir_name not in names:
            names.append(dir_name)
    return names


if __name__ == "__main__":
    txt_file = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand_900hotdog__.txt"
    names = dir_name(txt_file)
    for n in names:
        print(n)
