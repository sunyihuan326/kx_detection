#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 
# @Author  : sunyihuan
'''
某一文件夹按比例，分为train、test，并将文件名称写入到xxx_train.txt、xxx_test.txt文件中
'''
import os
import random

root_path = "H:/Joyoung/WLS/KX_FOODSets_model_data"


# 文件根本目录
# 目录中含有的文件如下：JPGImages
#                       Annotations
#                       ImageSets
#                            Main
#                                test.txt
#                                train.txt
#        其中，JPGImages目录下为所有jpg图片文件
#              Annotations目录下为所有xml标注文件
#              test.txt为所有的test集图片name（不含.jpg）
#              train.txt为所有的train集图片name（不含.jpg）

def split_data(clasees, train_percent):
    '''
    按类别名称将每一类分为test、train
    :param clasees: 类别名称，格式为list
    :param train_percent: train集的占比，一般为0.7-0.9
    :return:
    '''
    if not os.path.exists(root_path):
        print("cannot find such directory: " + root_path)
        exit()
    xmlfilepath = root_path + '/Annotations' + "/{}".format(clasees)
    txtsavepath = root_path + '/ImageSets/Main'

    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tr = int(num * train_percent)
    train = random.sample(list, tr)

    print("train size:", tr)
    ftest = open(txtsavepath + '/{}_test.txt'.format(str(clasees).lower()), 'w')
    ftrain = open(txtsavepath + '/{}_train.txt'.format(str(clasees).lower()), 'w')

    for i in list:
        if str(total_xml[i]).endswith("xml"):
            name = total_xml[i][:-4] + '\n'
            if i in train:
                ftrain.write(name)
            else:
                ftest.write(name)

    ftrain.close()
    ftest.close()


if __name__ == "__main__":
    clasees = ["BeefSteak", "CartoonCookies", "ChickenWings", "ChiffonCake", "Cookies",
               "CranberryCookies", "CupCake", "EggTart", "nofood", "Peanuts",
               "Pizza", "PorkChops", "PurpleSweetPotato", "RoastedChicken", "Toast"]
    print(len(clasees))
    train_percent = 0.8
    for c in clasees:
        split_data(c, train_percent)
