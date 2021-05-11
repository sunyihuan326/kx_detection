# -*- coding: utf-8 -*-
# @Time    : 2021/4/14
# @Author  : sunyihuan
# @File    : copy_img_from_brightness.py
'''
从excel中读取数据，并拷贝一定亮度条件下的数据

'''
import os
import shutil
import xlrd

excel_save = "F:/model_data/ZG/Li/vocleddata-food38-20210118/train/brightness.xls"
save_dir = "F:/model_data/ZG/Li/vocleddata-food38-20210118/train/bri"
if not os.path.exists(save_dir): os.mkdir(save_dir)
excel = xlrd.open_workbook(excel_save)
sheet = excel.sheet_by_index(0)
for kk in range(1, sheet.nrows):
    jpg_path = sheet.cell(kk, 0).value
    # print(sheet.cell(kk, 1).value)
    try:
        b = float(sheet.cell(kk, 1).value)
        if b > 150:
            shutil.copy(jpg_path, save_dir + "/" + jpg_path.split("/")[-1])
    except:
        print(jpg_path)
