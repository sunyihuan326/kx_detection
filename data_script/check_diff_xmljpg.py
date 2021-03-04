# -*- coding: utf-8 -*-
# @Time    : 2020/4/24
# @Author  : sunyihuan
# @File    : check_diff_xmljpg.py

# 输出两个文件中不同的名称（xml、jpg文件）
# 输出jpg、layer数据的不同

import os


def check_xmljpg_diff(img_dir, xml_dir):
    '''
    输出两个文件中不同的名字
    :param img_dir: 图片地址
    :param xml_dir: xml文件标注地址
    :return:
    '''
    xml_name_list = [o.split(".")[0] for o in os.listdir(xml_dir)]
    img_name_list = [o.split(".")[0] for o in os.listdir(img_dir)]

    # jpg中有,xml中没有
    print("图片总数：", len(img_name_list))
    print("未标注图片名称：")
    for i in img_name_list:
        if i not in xml_name_list:
            print(i)

    # xml中有，jpg中没有的
    print("已标注总数：", len(xml_name_list))
    print("已标注，但图片已删除名称：")
    for i in xml_name_list:
        if i not in img_name_list:
            print(i)


def check_layerjpg_diff(img_dir, layer_dir):
    '''
    输出两个文件中不同的名字
    :param img_dir: 图片地址
    :param layer_dir: layer数据地址
    :return:
    '''
    layer_name_list = []
    for b in os.listdir(layer_dir):
        if b in ["bottom", "middle", "top", "others"]:
            if os.path.exists(layer_dir + "/" + b):
                for jpg in os.listdir(layer_dir + "/" + b):
                    layer_name_list.append(jpg.split(".")[0])

    img_name_list = [o.split(".")[0] for o in os.listdir(img_dir)]

    # jpg中有,layer中没有
    print("图片总数：", len(img_name_list))
    print("未分类图片名称：")
    for i in img_name_list:
        if i not in layer_name_list:
            print(i)

    # layer中有，jpg中没有的
    print("已分类总数：", len(layer_name_list))
    print("已分类，但图片已删除名称：")
    for i in layer_name_list:
        if i not in img_name_list:
            print(i)


if __name__ == "__main__":
    img_dir = "E:/已标数据备份/X新模组数据/X/JPGImages"
    xml_dir = "E:/已标数据备份/X新模组数据/X/Annotations"
    layer_dir = "E:/已标数据备份/X新模组数据/X/layer_data"
    check_xmljpg_diff(img_dir, xml_dir)
    check_layerjpg_diff(img_dir, layer_dir)
