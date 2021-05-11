# -*- coding: utf-8 -*-
# @Time    : 2021/3/29
# @Author  : sunyihuan
# @File    : all_nums_from_results.py
'''
根据分类后的文件夹，输出对应结果
'''

import xlwt
import os


def results_all_nums(data_root):
    cls_list = os.listdir(data_root)
    w = xlwt.Workbook()
    sheet = w.add_sheet("all_nums")
    sheet.write(0, 0, "类别")
    sheet.write(0, 1, "高置信度准确数")
    sheet.write(0, 2, "低置信度准确数")
    sheet.write(0, 3, "无任何结果")
    sheet.write(0, 4, "高置信度错误")
    sheet.write(0, 5, "低置信度错误")
    sheet.write(0, 6, "错误无任何结果")
    for i, c in enumerate(cls_list):
        class_gao_nums = 0
        class_di_nums = 0
        class_noresult = 0
        class_error_gao_nums = 0
        class_error_di_nums = 0
        class_error_noresults_nums = 0
        if len(c.split(".")) == 1:
            class_name = c
            for pre_c in os.listdir(data_root + "/" + c):
                if pre_c == c:
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/gaofen"):
                        class_gao_nums += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/gaofen"))
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/difen"):
                        class_di_nums += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/difen"))
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/noresult"):
                        class_noresult += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/noresult"))
                elif pre_c == "noresult":
                    class_noresult += len(os.listdir(data_root + "/" + c + "/" + pre_c))
                else:
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/gaofen"):
                        class_error_gao_nums += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/gaofen"))
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/difen"):
                        class_error_di_nums += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/difen"))
                    if os.path.exists(data_root + "/" + c + "/" + pre_c + "/noresult"):
                        class_error_noresults_nums += len(os.listdir(data_root + "/" + c + "/" + pre_c + "/noresult"))
        sheet.write(i + 1, 0, c)
        sheet.write(i + 1, 1, class_gao_nums)
        sheet.write(i + 1, 2, class_di_nums)
        sheet.write(i + 1, 3, class_noresult)
        sheet.write(i + 1, 4, class_error_gao_nums)
        sheet.write(i + 1, 5, class_error_di_nums)
        sheet.write(i + 1, 6, class_error_noresults_nums)
    w.save(data_root + "/all_nums.xls")


data_root = "F:/serve_data/202101-03_detetction"
results_all_nums(data_root)
