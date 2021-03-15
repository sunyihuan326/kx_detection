# -*- coding: utf-8 -*-
# @Time    : 2021/3/3
# @Author  : sunyihuan
# @File    : all2classses.py
'''
按标签类别，将图片和xml文件分到对应文件夹
'''
import os
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET


def get_label_name(file):
    t_list = []
    if file.endswith('xml'):
        tree = ET.parse(file)
        root = tree.getroot()
        for object1 in root.findall('object'):
            for sku in object1.findall('name'):
                t_list.append(sku.text.lower())
    return t_list


def split_jpg_xml(file_root):
    jpg_root = file_root + "/" + "JPGImages"
    xml_root = file_root + "/" + "Annotations"
    jpg_len = len(os.listdir(jpg_root))
    xml_len = len(os.listdir(xml_root))
    print("图片总数：", jpg_len)
    print("xml文件总数：", xml_len)
    for j in tqdm(os.listdir(jpg_root)):
        jpg_name = jpg_root + "/" + j  # 图片全路径
        xml_ = j.split(".jpg")[0] + ".xml"
        xml_name = xml_root + "/" + xml_  # xml文件全路径
        if os.path.exists(xml_name):
            labels_l = get_label_name(xml_name)
            if len(list(set(labels_l))) == 1:
                jpg_cls_dir = jpg_root + "/" + labels_l[0]  # 单类图片文件夹地址
                xml_cls_dir = xml_root + "/" + labels_l[0]  # 单类xml文件夹地址
                if not os.path.exists(jpg_cls_dir): os.mkdir(jpg_cls_dir)
                if not os.path.exists(xml_cls_dir): os.mkdir(xml_cls_dir)
                shutil.move(jpg_name, jpg_cls_dir + "/" + j)
                shutil.move(xml_name, xml_cls_dir + "/" + xml_)
        else:
            no_xml_jpg = jpg_root + "/" + "no_xml_jpg"
            if not os.path.exists(no_xml_jpg): os.mkdir(no_xml_jpg)
            shutil.move(jpg_name, no_xml_jpg + "/" + j)


if __name__ == '__main__':
    file_root = "E:/已标数据备份/二期数据/bu/20200924"
    split_jpg_xml(file_root)
