# -*- coding: utf-8 -*-
# @Time    : 2020/3/6
# @Author  : sunyihuan
# @File    : from_jpgname_split_layer.py

'''
将img文件中图片按图片名称中的烤层信息分类到对应的烤层文件夹
'''

import os
import shutil


def copy_layer_jpgs(img_dir, layer_dir):
    '''
    将img文件中图片按图片名称中的烤层标签分类到对应的烤层文件夹
    :param img_dir: 图片文件夹
    :param layer_dir: 烤层分类文件夹
    :return:
    '''
    bottom_dir = layer_dir + "/bottom"  # 底层
    if not os.path.exists(bottom_dir): os.mkdir(bottom_dir)

    middle_dir = layer_dir + "/middle"  # 中层
    if not os.path.exists(middle_dir): os.mkdir(middle_dir)

    top_dir = layer_dir + "/top"  # 上层
    if not os.path.exists(top_dir): os.mkdir(top_dir)

    others_dir = layer_dir + "/others"  # 其他
    if not os.path.exists(others_dir): os.mkdir(others_dir)

    for jpgname in os.listdir(img_dir):
        if jpgname.endswith(".jpg"):
            if "shang" in jpgname:  # “shang” 上层标签
                shutil.copy(img_dir + "/" + jpgname, top_dir + "/" + jpgname)
            elif "zhong" in jpgname:  # “zhong” 中层标签
                shutil.copy(img_dir + "/" + jpgname, middle_dir + "/" + jpgname)
            elif "xia" in jpgname:  # “xia” 下层标签
                shutil.copy(img_dir + "/" + jpgname, bottom_dir + "/" + jpgname)
            else:
                print("error")


if __name__ == "__main__":
    img_dir = "E:/已标数据备份/二期数据/第一批/X5_20200310/chestnut"
    layer_dir = "E:/已标数据备份/二期数据/第一批/X5_20200310/layer_data/chestnut"
    if not os.path.exists(layer_dir): os.mkdir(layer_dir)
    copy_layer_jpgs(img_dir, layer_dir)
