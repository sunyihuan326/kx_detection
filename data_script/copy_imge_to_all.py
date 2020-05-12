# -*- coding: utf-8 -*-
# @Time    : 202003/2/29
# @Author  : sunyihuan
# @File    : copy_imge_to_all.py
'''
将图片从子文件夹中拷贝到根目录下

'''


import shutil
import os

img_dir = "E:/WLS_originalData/二期数据/第一批/X5_20200310/chestnut"

dst_img_dir = "E:/WLS_originalData/二期数据/第一批/X5_20200310/chestnut"
for k in ["kaojia", "kaojia(bujiaxizhi)", "kaopan", "kaopan(bujiaxizhi)"]:
    img_dirs = dst_img_dir + "/" + k
    for l in ["shang", "xia", "zhong"]:
        img_dir_name = img_dirs + "/" + l
        for jpg_ in os.listdir(img_dir_name):
            jpg_name = img_dir_name + "/" + jpg_

            dst_jpg_name = dst_img_dir + "/" + jpg_

            print(jpg_name)
            print(dst_jpg_name)
            shutil.copy(jpg_name, dst_jpg_name)
