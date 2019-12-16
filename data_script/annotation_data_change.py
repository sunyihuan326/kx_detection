# -*- encoding: utf-8 -*-

"""
@File    : annotation_data_change.py
@Time    : 2019/12/13 11:17
@Author  : sunyihuan
"""

import os

right_txt_path = "E:/kx_detection/multi_detection/data/dataset/20191206/train1206.txt"  # 正确标签txt文件
right_txt_lines = open(right_txt_path, "r").readlines()
right_list = [l.strip() for l in right_txt_lines]

dst_txt_path = "E:/kx_detection/multi_detection/data/dataset/XandOld/train0926_oldAndX1206.txt"  # 目标txt文件
dst_txt_lines = open(dst_txt_path, "r").readlines()
dst_list = [l.strip() for l in dst_txt_lines]

new_dst_list = []
for rl in right_list:
    for dl in dst_list:
        if dl.split(".jpg")[0] == rl.split(".jpg")[0]:
            dl_new = dl.split(".jpg")[0] + ".jpg" + dl.split(".jpg")[1]
            new_dst_list.append(dl_new)
            print(dl.split(".jpg")[0])
        else:
            new_dst_list.append(dl)

new_txt_name = "E:/kx_detection/multi_detection/data/dataset/XandOld/train0926_oldAndX1206_new.txt"
file = open(new_txt_name, "w")
for i in new_dst_list:
    file.write(i)