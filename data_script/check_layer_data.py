# -*- encoding: utf-8 -*-

"""
查看layer数据是否有漏标注
@File    : check_layer_data.py
@Time    : 2019/10/25 10:51
@Author  : sunyihuan
"""
import os


def get_layer_data(layer_root):
    '''
    获取所有的layer数据
    :param layer_root: layer文件目录
                           格式为：layer_root
                                         bottom
                                         middle
                                         others
                                         top
    :return:
    '''
    layer_list = []
    for c in ["bottom", "middle", "top", "others"]:
        c_dir = layer_root + "/" + c
        for img in os.listdir(c_dir):
            layer_list.append(img)
    return layer_list


def check(orignal_img_dir, layer_root):
    '''
    检测遗漏的文件，并统计数量
    :param orignal_img_dir: 原始图片文件夹地址
    :param layer_root:  layer文件目录
    :return:
    '''
    o_img_list = os.listdir(orignal_img_dir)
    layer_list = get_layer_data(layer_root)
    print(layer_list)
    cou = 0
    for i in o_img_list:
        if i not in layer_list:
            cou += 1
            print(i)
    return cou


orignal_img_dir = "E:/DataSets/KX_FOODSets_model_data/X_27classes_1025/JPGImages"  # 原图片文件夹地址
layer_root = "E:/layer_data/X_KX_data_27classes1025"  # layer数据根目录
print(check(orignal_img_dir, layer_root))
