# -*- coding: utf-8 -*-
# @Time    : 2020/9/8
# @Author  : sunyihuan
# @File    : from_xml_rename_jpg.py
'''
根据xml文件中object标签框名称，修改jpg文件名
'''

import os
import xml.etree.ElementTree as ET


def change_jpg_name(xml_dir, img_dir):
    '''
    检查标签是否有更改完成，标签名不正确的输出
    :param inputpath: xml文件路径
    :param label_name: 标签名
    :return:
    '''
    listdir = os.listdir(xml_dir)
    for file0 in listdir:
        if file0.endswith('xml'):
            file = os.path.join(xml_dir, file0)
            tree = ET.parse(file)
            root = tree.getroot()
            if "corncut" in file:
                ty = False
                for object1 in root.findall('object'):
                    for sku in object1.findall('name'):
                        if sku.text == "cornone":
                            ty = True
                if ty:
                    os.rename(file, file.replace("corncut", "___cornone"))  # 修改xml文件名
                    # jpg_name = file0.split(".xml")[0] + ".jpg"
                    # os.rename(img_dir + "/" + jpg_name, img_dir + "/" + jpg_name.replace("corncut", "___cornone"))  # 修改jpg文件名


if __name__ == "__main__":
    xml_dir = "F:/model_data/ZG/SF1/202008/Annotations"
    img_dir = "F:/model_data/ZG/SF1/202008/JPGImages"
    change_jpg_name(xml_dir, img_dir)
