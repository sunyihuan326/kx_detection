# -*- coding: utf-8 -*-
# @Time    : 2020/11/27
# @Author  : sunyihuan
# @File    : two_txt_sub.py
'''
查看两个txt文件的差
'''
import os


def two_txt_sub(txt1, txt2):
    t = []
    txt1_list = open(txt1, "r").readlines()
    txt2_list = open(txt2, "r").readlines()
    img1_list = [c.split(".jpg")[0].split("/")[-1] for c in txt1_list]
    img2_list = [c.split(".jpg")[0].split("/")[-1] for c in txt2_list]
    sub_n = 0
    print(len(set(img1_list)|set(img2_list)))
    print(len(img1_list))
    print(len(set(img2_list)))
    print(len(img2_list))
    # for k in img2_list:
    #     if k not in img1_list:
    #         sub_n += 1
    #         print(k)
    return sub_n


if __name__ == "__main__":
    txt1 = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand_900hotdog.txt"
    txt2 = "E:/ckpt_dirs/Food_detection/multi_food5/serve_3660train39_hot_zi_lv_strand20201120.txt"
    sub_n = two_txt_sub(txt1, txt2)
    print(sub_n)
