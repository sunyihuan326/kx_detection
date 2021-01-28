# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : write_classes_nums_from_txt.py
'''
从生成的txt文件中，输出各类的数量
从生成的txt文件中，输出各类检测框的数量

'''
import xlwt
import xlrd

id_2_name = {0: "牛排", 1: "卡通饼干", 2: "鸡翅", 3: "戚风蛋糕", 4: "戚风蛋糕", 5: "曲奇饼干"
    , 6: "蔓越莓饼干", 7: "纸杯蛋糕", 8: "蛋挞", 9: "空", 10: "花生米"
    , 11: "披萨", 12: "披萨", 13: "披萨", 14: "排骨", 15: "土豆切"
    , 16: "大土豆", 17: "小土豆", 18: "红薯切", 19: "大红薯", 20: "小红薯"
    , 21: "烤鸡", 22: "吐司", 23: "板栗", 24: "玉米", 25: "玉米"
    , 26: "鸡腿", 27: "芋头", 28: "小馒头", 29: "整个茄子", 30: "切开茄子"
    , 31: "吐司面包", 32: "餐具", 33: "餐具", 34: "鱼", 35: "热狗"
    , 36: "虾", 37: "虾", 38: "烤肉串", 39: "锡纸", 101: "戚风蛋糕"
    , 40: "大土豆", 41: "大红薯"}


def cls_nums(txt_path):
    '''
    输出各类别数量
    :param txt_path:
    :return:
    '''
    nums_dict = {}
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    for tt in txt_files:
        tt = tt.strip()
        cls = tt.split(" ")[-1].split(",")[-1]
        if cls not in nums_dict.keys():
            nums_dict[cls] = 1
        else:
            nums_dict[cls] += 1
    return nums_dict


def cls_bboxes_nums(txt_path):
    '''
    输出各类别检测框数量
    :param txt_path:
    :return:
    '''
    nums_dict = {}
    txt_file = open(txt_path, "r")
    txt_files = txt_file.readlines()
    for tt in txt_files:
        tt = tt.strip()
        bboxes_all = tt.split(" ")[2:]
        for kk in bboxes_all:
            cls = kk.split(",")[-1]
            if cls not in nums_dict.keys():
                nums_dict[cls] = 1
            else:
                nums_dict[cls] += 1
    return nums_dict


if __name__ == "__main__":
    txt_path = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand20210115.txt"
    w = xlwt.Workbook()
    sheet = w.add_sheet("cls_nums")
    sheet.write(0, 0, "classes")
    sheet.write(0, 1, "类别数量")
    sheet.write(0, 2, "检测框数量")
    cls_nums_dict = cls_nums(txt_path)
    cls_bboxes_nums_dict = cls_bboxes_nums(txt_path)
    i = 1
    for cc in cls_nums_dict.keys():
        sheet.write(i, 0, id_2_name[int(cc)])
        sheet.write(i, 1, cls_nums_dict[cc])
        sheet.write(i, 2, cls_bboxes_nums_dict[cc])
        i += 1
    w.save("cls_nums.xls")
