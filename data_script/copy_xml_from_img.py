# -*- encoding: utf-8 -*-

"""
根据img文件移动xml文件
@File    : copy_xml_from_img.py
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
                print(img_xml_name)
                shutil.move(xml_dir + "/" + img_xml_name, xml_save_dir + "/" + img_xml_name)


if __name__ == "__main__":
    img_root = "F:/serve_data/202101-03formodel/exrtact_file/JPGImages/nouse"
    # img_dir = "C:/Users/sunyihuan/Desktop/pizza_img/pizza_four"
    xml_root = "F:/serve_data/202101-03formodel/exrtact_file/Annotations"

    xml_save_root = "F:/serve_data/202101-03formodel/exrtact_file/Annotations/nouse"
    if not os.path.exists(xml_save_root): os.mkdir(xml_save_root)
    for c in os.listdir(img_root):
        img_dir = img_root + "/" + c
        xml_dir = xml_root + "/" + c

        xml_save_dir = xml_save_root + "/" + c
        if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)

        copy_xml(img_dir, xml_dir, xml_save_dir)
