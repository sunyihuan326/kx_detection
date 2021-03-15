# -*- coding: utf-8 -*-
# @Time    : 2021/3/9
# @Author  : sunyihuan
# @File    : all_layer2classes_layer.py
'''
将bottom、middle、top、others下所有图片，按JPGImages中各类文件夹，单独分为：xxxx
                                                                           bottom
                                                                           middle
                                                                           top
                                                                           others
layer_data结构
             bottom
             middle
             top
             others
JPGImages结构：
           JPGImages
                xxxx1
                xxxx2
                xxxx3
                ……
目标layer_data结构：
           xxxx1
              bottom
              middle
              top
              others
           xxxx1
              bottom
              middle
              top
              others
'''

import os
import shutil
from tqdm import tqdm


def get_layer_list(layer_root):
    '''
    获取layer数据的列表，
    :param layer_root:
    :return:
    '''
    l = {}
    for ll in os.listdir(layer_root):
        l[ll] = os.listdir(layer_root + "/" + ll)
    return l


def split_layer_data(JPGImages_root, layer_root, new_layer_root):
    class_list = os.listdir(JPGImages_root)
    # 获取layer数据列表
    layer_dict = get_layer_list(layer_root)
    for c in class_list:
        cls_dir = JPGImages_root + "/" + c  # 单类文件夹根目录
        new_c_layer = new_layer_root + "/" + c  # 单类layer根目录
        if not os.path.exists(new_c_layer): os.mkdir(new_c_layer)
        for jpg in tqdm(os.listdir(cls_dir)):
            for l_d in layer_dict.keys():
                l_list = layer_dict[l_d]
                if jpg in l_list:
                    if not os.path.exists(new_c_layer + "/" + l_d): os.mkdir(new_c_layer + "/" + l_d)
                    shutil.move(cls_dir + "/" + jpg, new_c_layer + "/" + l_d + "/" + jpg)


if __name__ == "__main__":
    JPGImages_root = "E:/t/JPGImages"
    layer_root = "E:/t/layer_data"
    new_layer_root = "E:/t/layer_data_new"
    split_layer_data(JPGImages_root, layer_root, new_layer_root)
