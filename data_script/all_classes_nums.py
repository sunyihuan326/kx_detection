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
        if not k.endswith("DS_Store") and not k.endswith("xls"):
            c_nums = len(os.listdir(img_dirs + "/" + k))
            all_nums[k] = c_nums
    # all_nums=sorted(all_nums.items(), key=lambda item: item[1], reverse=True)
    return all_nums


if __name__ == "__main__":
    w = xlwt.Workbook()
    sheet0 = w.add_sheet("all_nums")
    dir_root = "E:/已标数据备份/二期数据/JPGImages"
    all_nums = print_nums(dir_root)
    sheet0.write(0, 0, "classes")
    sheet0.write(0, 1, "nums")
    c_c = 0
    for c in all_nums.keys():
        sheet0.write(c_c + 1, 0, c)
        sheet0.write(c_c + 1, 1, str(all_nums[c]))
        c_c += 1
    w.save("E:/已标数据备份/二期数据/JPGImages/all.xls")