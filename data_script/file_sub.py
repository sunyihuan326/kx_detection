#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 
# @Author  : sunyihuan

'''
A为所有文件，将不在B中的文件拷贝至C
'''
import os
import shutil
from tqdm import tqdm


def file_sub(all_data, using_dir, sub_save_dir):
    '''
    将在all_data文件夹中，但不在using_dir文件夹中的文件，保存到sub_save_dir中
    :param all_data: 所有图片地址
    :param using_dir: 已经使用图片地址
    :param sub_save_dir: 未使用图片要保存的地址
    :return:
    '''
    file_dirs = os.listdir(all_data)
    using_files = os.listdir(using_dir)
    all_use_f = []  # 所有已用列表
    for c in using_files:
        for f in os.listdir(os.path.join(using_dir, c)):
            all_use_f.append(f.strip())
    print(len(all_use_f))
    for fil in tqdm(file_dirs):
        if fil != ".DS_Store":
            if fil not in all_use_f:
                shutil.move(os.path.join(all_data, fil), os.path.join(sub_save_dir, fil))


if __name__ == "__main__":
    all_data = "F:/serve_data/202101-04/covert_jpg"
    using_dir = "F:/serve_data/202101-04/classes"
    sub_save_dir = "F:/serve_data/202101-04/classes_others"
    if not os.path.exists(sub_save_dir): os.mkdir(sub_save_dir)
    file_sub(all_data, using_dir, sub_save_dir)
