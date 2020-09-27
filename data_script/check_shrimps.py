# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : check_shrimps.py

txt_path = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand.txt"
txt_file = open(txt_path, "r")
txt_files = txt_file.readlines()
strand_c=0
_c=0
for cc in txt_files:
    cc =cc.strip()

    if "strand" in cc:
        strand_c+=1
    if int(cc.split(" ")[-1].split(",")[-1])==38:
        _c+=1
print(strand_c,_c)