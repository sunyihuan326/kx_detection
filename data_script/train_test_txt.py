#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 
# @Author  : sunyihuan
'''
某一文件夹按比例，分为train、test，并将文件名称写入到xxx_train.txt、xxx_test.txt、val.txt文件中
'''
import os
import random

root_path = "E:/已标数据备份/X补采"


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

def split_data(clasees, train_percent, test_percent):
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
    random.shuffle(total_xml)  # 打乱total_xml
    num = len(total_xml)
    list = range(num)
    tr = int(num * train_percent)  # train集数量
    train = total_xml[:tr]  # train集列表数据内容

    te = int(num * test_percent)  # test集数量
    test = total_xml[tr:tr + te]  # test集列表数据内容

    val = num - tr - te  # val集数量
    val_set = total_xml[tr + te:]  # val集列表数据内容

    print("train size:", tr)
    print("test size:", te)
    print("val size:", val)
    ftest = open(txtsavepath + '/{}_test.txt'.format(str(clasees).lower()), 'w')
    ftrain = open(txtsavepath + '/{}_train.txt'.format(str(clasees).lower()), 'w')
    fval = open(txtsavepath + '/{}_val.txt'.format(str(clasees).lower()), 'w')

    for x in total_xml:
        if str(x).endswith("xml"):
            name = x[:-4] + '\n'
            if x in train:
                ftrain.write(name)
            elif x in test:
                ftest.write(name)
            else:
                fval.write(name)

    ftrain.close()
    ftest.close()
    fval.close()


if __name__ == "__main__":
    # clasees = ["BeefSteak", "CartoonCookies", "ChickenWings", "ChiffonCake6", "ChiffonCake8",
    #            "Cookies", "CranberryCookies", "CupCake", "EggTart", "EggTartBig",
    #            "nofood", "Peanuts", "Pizzafour", "Pizzaone", "Pizzasix",
    #            "Pizzatwo", "PorkChops", "PotatoCut", "Potatol", "Potatom",
    #            "Potatos", "SweetPotatoCut", "SweetPotatol", "SweetPotatom", "SweetPotatos",
    #            "RoastedChicken", "Toast"]
    clasees = ["CartoonCookies", "Cookies", "CupCake", "Pizzafour", "Pizzaone",
               "Pizzasix", "Pizzatwo", "SweetPotatoS", "Toast"]
    print(len(clasees))
    train_percent = 0.8
    test_percent = 0.1
    for c in clasees:
        print(c)
        split_data(c, train_percent, test_percent)
