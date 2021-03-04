# -*- coding: utf-8 -*-
# @Time    : 2021/3/2
# @Author  : sunyihuan
# @File    : print_all_labels.py
'''
输出标注文件夹下，所有xml文件的标签

'''

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def label_name(file):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
    :return:
    '''
    t_list = []
    if file.endswith('xml'):
        tree = ET.parse(file)
        root = tree.getroot()
        for object1 in root.findall('object'):
            for sku in object1.findall('name'):
                t_list.append(sku.text.lower())
    return t_list


if __name__ == "__main__":
    inpu_path = "E:/DataSets/2020_two_phase_KXData/only2phase_data/Annotations"
    xml_list = os.listdir(inpu_path)
    all_labels = []
    for xml in tqdm(xml_list):
        xml_file = inpu_path + "/" + xml
        t_list = label_name(xml_file)
        if len(t_list) > 0:
            for t_l in t_list:
                all_labels.append(t_l)
    all_labels = list(set(all_labels))
    print("总标签数：", len(all_labels))
    print("标签名字", all_labels)
