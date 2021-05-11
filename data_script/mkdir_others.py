# -*- coding: utf-8 -*-
# @Time    : 2021/4/15
# @Author  : sunyihuan
# @File    : mkdir_others.py

'''
如果文件中无others文件，创建
'''
import os
import shutil
from tqdm import tqdm


def mk_dir(img_dir):
    if "others" not in os.listdir(img_dir):
        os.mkdir(img_dir + "/others")


if __name__ == "__main__":
    img_root = "F:/Test_set/OVEN/JPGImages"
    for c in tqdm(os.listdir(img_root)):
        img_dir = img_root + "/" + c
        mk_dir(img_dir)
