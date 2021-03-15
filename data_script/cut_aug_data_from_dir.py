# -*- coding: utf-8 -*-
# @Time    : 2021/3/4
# @Author  : sunyihuan
# @File    : cut_aug_data_from_dir.py
'''
移除文件夹下所有增强数据

'''

import os
import shutil
from tqdm import tqdm


def cut_aug_data(file_root):
    '''

    :param file_root: 文件根目录，文件格式为：
                    file_root
                         Annotations
                         JPGImages
                         layer_data
                    将文件名中带有aug标签（如_hot.jpg）的数据移动至aug_data文件下
    :return:
    '''
    cut_root = file_root + "/aug_data"
    if not os.path.exists(cut_root): os.mkdir(cut_root)

    for r in ["Annotations", "JPGImages", "layer_data"]:
        root_r = file_root + "/" + r
        aug_r = cut_root + "/" + r
        if not os.path.exists(aug_r): os.mkdir(aug_r)

        for c in tqdm(os.listdir(root_r)):
            c_dir = root_r + "/" + c
            for f in os.listdir(c_dir):
                aug_c_fir = aug_r + "/" + c
                if not os.path.exists(aug_c_fir): os.mkdir(aug_c_fir)
                if r == "layer_data":
                    print(c_dir + "/" + f)
                    for b in os.listdir(c_dir + "/" + f):
                        aug_b_dir = aug_c_fir + "/" + f
                        if not os.path.exists(aug_b_dir): os.mkdir(aug_b_dir)
                        if "_hot.jpg" in b or "_huang.jpg" in b or "_lv.jpg" in b or "_hong.jpg" in b or "_zi.jpg" in b \
                                or "_hot.xml" in b or "_huang.xml" in b or "_lv.xml" in b or "_hong.xml" in b or "_zi.xml" in b:
                            fname = c_dir + "/" + f + "/" + b
                            shutil.move(fname, aug_b_dir + "/" + f)
                else:
                    if "_hot.jpg" in f or "_huang.jpg" in f or "_lv.jpg" in f or "_hong.jpg" in f or "_zi.jpg" in f \
                            or "_hot.xml" in f or "_huang.xml" in f or "_lv.xml" in f or "_hong.xml" in f or "_zi.xml" in f:
                        fname = c_dir + "/" + f
                        shutil.move(fname, aug_c_fir + "/" + f)


if __name__ == "__main__":
    file_root = "E:/已标数据备份/二期数据(无serve_data)2020"
    cut_aug_data(file_root)
