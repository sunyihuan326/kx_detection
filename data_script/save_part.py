# -*- coding: utf-8 -*-
# @Time    : 2020/4/26
# @Author  : sunyihuan
# @File    : save_part.py

# 按某一文件下保存部分，其余的删除

import os
import shutil


def save_part(all_data_dir, part_data_dir):
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