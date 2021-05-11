# -*- coding: utf-8 -*-
# @Time    : 2021/4/28
# @Author  : sunyihuan
# @File    : split_cls_from_dir.py

"""
A为所有文件集合，且A中按小类别分类
将B中文件，按A标准分至对应文件夹
A：peanuts
      xxxx.jpg
   corn
      xxxx.jpg
      ……
B：xxx.jpg
   ……

目标格式：  B:peanuts
                 xxxx.jpg
             corn
                 xxxx.jpg
              ……
"""
import os
import shutil
from tqdm import tqdm

a_dir = "F:/serve_data/ZG_data/2021all_data"
b_dir = "F:/serve_data/ZG_data/20210129/2021_noresults"

b_file_list = os.listdir(b_dir)

cc_d = {}
for cc in os.listdir(a_dir):
    cc_d[cc] = os.listdir(a_dir + "/" + cc)

for f in tqdm(b_file_list):
    if f.endswith(".jpg"):
        for cc in cc_d.keys():
            if f in cc_d[cc]:
                if not os.path.exists(b_dir + "/" + cc): os.mkdir(b_dir + "/" + cc)
                shutil.move(b_dir + "/" + f, b_dir + "/" + cc + "/" + f)
                continue
