# -*- coding: utf-8 -*-
# @Time    : 2020/8/13
# @Author  : sunyihuan
# @File    : all_classes_nums.py
'''
输出文件夹下各类别种类数
'''

import os
import xlwt


def print_nums(img_dirs):
    all_nums = {}
    for k in os.listdir(img_dirs):
        if not k.endswith("DS_Store"):
            c_nums = len(os.listdir(img_dirs + "/" + k))
            all_nums[k] = c_nums
    # all_nums=sorted(all_nums.items(), key=lambda item: item[1], reverse=True)
    return all_nums


if __name__ == "__main__":
    w = xlwt.Workbook()
    sheet0 = w.add_sheet("all_nums")
    dir_root = "C:/Users/sunyihuan/Desktop/test_img/t0"
    all_nums = print_nums(dir_root)
    sheet0.write(0, 0, "classes")
    sheet0.write(0, 1, "nums")
    c_c = 0
    for c in all_nums.keys():
        sheet0.write(c_c + 1, 0, c)
        sheet0.write(c_c + 1, 1, str(all_nums[c]))
        c_c += 1
    w.save("C:/Users/sunyihuan/Desktop/test_img/t0/all.xls")