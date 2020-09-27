# -*- coding: utf-8 -*-
# @Time    : 2020/8/13
# @Author  : sunyihuan
# @File    : all_classes_nums.py
'''
输出文件夹下各类别种类数
'''

import os


def print_nums(img_dirs):
    all_nums = {}
    for k in os.listdir(img_dirs):
        if not k.endswith("DS_Store"):
            c_nums = len(os.listdir(img_dirs + "/" + k))
            all_nums[k] = c_nums
    all_nums=sorted(all_nums.items(), key=lambda item: item[1], reverse=True)
    return all_nums


if __name__ == "__main__":
    dir_root = "E:/WLS_originalData/3660bucai(annotation)/JPGImages"
    print(print_nums(dir_root))
