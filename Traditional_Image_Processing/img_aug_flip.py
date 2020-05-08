# -*- encoding: utf-8 -*-

"""
图片翻转，包含xml文件处理

@File    : img_aug_flip.py
@Time    : 2019/11/22 10:05
@Author  : sunyihuan
"""
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import cv2


def img_flip(img_path, mode="x"):
    '''
    图片镜像翻转
    :param img: 图片地址
    :return:
    '''
    img1 = cv2.imread(img_path)  # 目标图片
    if mode == "x":
        img = cv2.flip(img1, 1)  # 水平翻转
    elif mode == "y":
        img = cv2.flip(img1, 0)  # 垂直翻转
    else:
        img = cv2.flip(img1, -1)  # 水平垂直翻转
    return img


def xml_flip(xml_path, mode, xml_save_path):
    '''
    xml文件中坐标位置修改
    :param xml_path: xml文件位置
    :param w:图片width
    :param h:图片higth
    :param xml_save_path:xml文件保存路径
    :return:
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            ymin = sku.find("ymin")
            ymax = sku.find("ymax")
            xmin = sku.find("xmin")
            xmax = sku.find("xmax")
            if mode == "x":
                xmin.text = str(max(800 - int(xmin.text), 0))
                xmax.text = str(min(800 - int(xmax.text), 800))
            elif mode == "y":
                ymin.text = str(max(600 - int(ymin.text), 0))
                ymax.text = str(min(600 - int(ymax.text), 600))
            else:
                xmin.text = str(max(800 - int(xmin.text), 0))
                xmax.text = str(min(800 - int(xmax.text), 800))
                ymin.text = str(max(600 - int(ymin.text), 0))
                ymax.text = str(min(600 - int(ymax.text), 600))

    tree.write(xml_save_path, encoding='utf-8')


def img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    文件夹padding和保存
    :param img_dir:图片文件夹地址
    :param xml_dir: xml文件夹地址
    :param img_save_dir:图片保存文件夹地址
    :param xml_save_dir: xml保存文件夹地址
    :return:
    '''
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith("jpg"):
            mode = "x"
            img_path = img_dir + "/" + img_file
            img = img_flip(img_path, mode)  # 图片翻转处理
            img_name = str(img_file).split(".")[0] + "_" + mode + ".jpg"  # 图片名称
            cv2.imwrite(img_save_dir + "/" + img_name, img)  # 图片保存
            xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
            xml_flip(xml_name, mode, xml_save_name)  # xml文件标注更改、保存


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/JPGImages"
    xml_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/flip_x"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/flip_x_annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
