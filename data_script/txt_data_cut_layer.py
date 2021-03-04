# -*- coding: utf-8 -*-
# @Time    : 2021/2/20
# @Author  : sunyihuan
# @File    : txt_data_cut_layer.py
'''
去除txt文件数据中，layer标签数据

'''

import os


def cut_data(txt_file, save_file):
    txt_file = open(txt_file, "r")
    txt_files = txt_file.readlines()
    print(len(txt_files))

    new_files = []
    for t in txt_files:
        t = t.strip()
        t_l = t.split(" ")
        jpg_name = t_l[0]
        data = t_l[2:]
        for d in data:
            jpg_name += " " + d

        new_files.append(jpg_name + "\n")
    file = open(save_file, "w")
    for i in new_files:
        file.write(i)


txt_file = "E:/DataSets/X_3660_data/test39.txt"
save_file = "test39.txt"
cut_data(txt_file, save_file)
