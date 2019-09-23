#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 
# @Author  : sunyihuan

'''
拷贝未使用的数据
'''
import os
import shutil


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
    for file_dir in file_dirs:
        if file_dir != ".DS_Store":
            file_name = os.listdir(os.path.join(all_data, file_dir))
            for file_n in file_name:
                if file_n != ".DS_Store":
                    if file_n not in using_files:
                        if not os.path.exists(os.path.join(sub_save_dir, file_dir)): os.mkdir(
                            os.path.join(sub_save_dir, file_dir))
                        save_file_name = os.path.join(sub_save_dir, file_dir) + "/" + file_n
                        print(save_file_name)
                        shutil.copy(os.path.join(os.path.join(all_data, file_dir), file_n), save_file_name)


if __name__ == "__main__":
    all_data = "E:/DataSets/KX_FOODSets/JPGImages"
    using_dir = "E:/DataSets/KX_FOODSets_model_data/15classes_0722/JPGImages"
    sub_save_dir = "E:/DataSets/已标注未使用数据/JPGImages"
    file_sub(all_data, using_dir, sub_save_dir)
