# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : sunyihuan
# @File    : change_xml_labels.py
'''
根据文件名，更改xml中标签数据

'''
import os
import xml.etree.ElementTree as ET


def check_labelname(inputpath):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
    :return:
    '''
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath, file)
            tree = ET.parse(file)
            root = tree.getroot()
            if "size4" in file:
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        sku.text = "chiffoncake6"
                        tree.write(file, encoding='utf-8')
            elif "size10" in file:
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        sku.text = "chiffoncake8"
                        tree.write(file, encoding='utf-8')
            else:
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        sku.text = "chiffoncake8"
                        tree.write(file, encoding='utf-8')


if __name__ == "__main__":
    inpu_path = "E:/DataSets/X_3660_data/bu/20200902/Annotations/chiffoncake"
    check_labelname(inpu_path)
