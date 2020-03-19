# -*- encoding: utf-8 -*-

"""
根据img文件拷贝xml文件
@File    : from_img_copy_xml.py
@Time    : 2019/12/3 8:45
@Author  : sunyihuan
"""
import os
import shutil


def copy_xml(img_dir, xml_dir, xml_save_dir):
    '''
    根据img_dir中文件名，拷贝xml文件至xml_save_dir
    :param img_dir: 图片地址
    :param xml_dir: 原xml文件地址
    :param xml_save_dir: 保存xml文件地址
    :return:
    '''
    xml_list = os.listdir(xml_dir)
    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg"):
            img_name0 = img_name.split(".jpg")[0]
            img_xml_name = img_name0 + ".xml"
            if img_xml_name in xml_list:
                shutil.copy(xml_dir + "/" + img_xml_name, xml_save_dir + "/" + img_xml_name)


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/pizza_img/pizza_four"
    xml_dir = "C:/Users/sunyihuan/Desktop/pizza_img/Annotations"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/pizza_img/pizza_annotations"
    copy_xml(img_dir, xml_dir, xml_save_dir)
