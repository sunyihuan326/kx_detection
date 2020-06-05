# -*- coding: utf-8 -*-
# @Time    : 2020/5/29
# @Author  : sunyihuan
# @File    : cut_txt_from_jpgdir.py
'''
从txt中删除某一文件夹下所有数据
'''
import os


def cut_txt(txt_path, txt_new_path, cut_list):
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()

    train_all_list = []
    for txt_f in txt_files:
        txt_f_name = txt_f.split("/")[-1].split(".jpg")[0]
        if txt_f_name not in cut_list:
            train_all_list.append(txt_f)

    file = open(txt_new_path, "w")
    for i in train_all_list:
        file.write(i)


def get_cut_list(jpg_dir):
    return [a.split(".")[0] for a in os.listdir(jpg_dir)]


if __name__ == "__main__":
    jpg_dir = "E:/zg_data/202005/JPGImages_false"
    cut_list = get_cut_list(jpg_dir)
    print(cut_list)
    txt_path = "E:/zg_data/202005/serve_test20.txt"
    txt_new_path = "E:/zg_data/202005/serve_test20_new.txt"
    cut_txt(txt_path, txt_new_path, cut_list)
