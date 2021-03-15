#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 
# @Author  : sunyihuan
'''
更改xml文件中filename的数据
主要是.jpg文件名称更改后，统一修改xml文件中的标注数据
'''
import os
import xml.etree.ElementTree as ET


def changeFilename(inputpath):
    '''
    更改xml文件中filename的数据
    :param inputpath: xml文件夹地址
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            filename = file.split(".")[0]
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for f in root.findall('filename'):
                f.text = filename + ".jpg"
                tree.write(file, encoding='utf-8')


if __name__ == '__main__':
    xml_root = "E:/DataSets/model_data/X_data2019/Annotations"
    for c in os.listdir(xml_root):
        inputpath = xml_root + "/" + c  # 这是xml文件的文件夹的绝对地址
        changeFilename(inputpath)
