# -*- coding: utf-8 -*-
# @Time    : 2020/8/26
# @Author  : sunyihuan
# @File    : write_dir_nums_to_excel.py
'''
将各文件数量写入到excel中
'''
import os
import xlwt

img_root = "E:/WLS_originalData/all_test_data/all_original_data_0910_error_detect5"
wb = xlwt.Workbook()
sh = wb.add_sheet("all_nums")
sh.write(0, 0, "name")
sh.write(0, 1, "nums")
for i, c in enumerate(os.listdir(img_root)):
    sh.write(i + 1, 0, c)
    sh.write(i + 1, 1, len(os.listdir(img_root + "/" + c)))
wb.save(img_root+"/nums.xls")
