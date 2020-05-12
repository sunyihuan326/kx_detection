# -*- coding: utf-8 -*-
# @Time    : 2020/4/26
# @Author  : sunyihuan
# @File    : save_part.py

# 按某一文件下保存部分，其余的删除.主要用于按照部分原图，保留layer数据

import os


def save_part(all_data_dir, part_data_dir):
    '''
    按照part_data_dir中的文件，将all_data_dir中多余的烤层分类数据删除

    :param all_data_dir: 所有数据地址，其中文件中包含：bottom、middle、top、others
    :param part_data_dir: 要保留的数据文件夹，如：hotdog_JPG未校准
    :return:
    '''
    p_list = [o.split(".")[0] for o in os.listdir(part_data_dir)]

    for b in os.listdir(all_data_dir):
        if b in ["bottom", "middle", "top", "others"]:
            for jpg in os.listdir(all_data_dir + "/" + b):
                if jpg.split(".")[0] not in p_list:
                    os.remove(all_data_dir + "/" + b + "/" + jpg)

if __name__ == "__main__":
    all_data_dir =  "E:/WLS_originalData/二期数据/第二批/不使用/hotdog_layer未校准"
    part_data_dir = "E:/WLS_originalData/二期数据/第二批/不使用/hotdog_JPG未校准"

    save_part(all_data_dir,part_data_dir)