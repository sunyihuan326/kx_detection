# -*- coding: utf-8 -*-
# @Time    : 2020/2/29
# @Author  : sunyihuan
# @File    : copy_imge.py


import shutil
import os

img_dir = "C:/Users/sunyihuan/Desktop/xiaomantou/hongtang"

dst_img_dir = "C:/Users/sunyihuan/Desktop/xiaomantou/hongtang"
for k in ["kaojia", "kaojia(bupuxizhi)", "kaopan", "kaopan(budaixizhi)"]:
    img_dirs = dst_img_dir + "/" + k
    for l in ["shang", "xia", "zhong"]:
        img_dir_name = img_dirs + "/" + l
        for jpg_ in os.listdir(img_dir_name):
            jpg_name = img_dir_name + "/" + jpg_

            dst_jpg_name = dst_img_dir + "/" + jpg_

            print(jpg_name)
            print(dst_jpg_name)
            shutil.copy(jpg_name, dst_jpg_name)
