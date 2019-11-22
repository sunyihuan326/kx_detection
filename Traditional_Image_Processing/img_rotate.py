# -*- encoding: utf-8 -*-

"""
@File    : img_rotate.py
@Time    : 2019/11/21 11:33
@Author  : sunyihuan
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import math
import random


def image_rotate(img_path, xml_path, xml_save_path):
    '''
    图片padding，将图照片自动padding成正方形
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    rota = random.randint(1, 3)
    img_new = img.rotate(rota, expand=1)  # 图片旋转

    # xml标注数据修改
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            ymin = sku.find("ymin")
            xmin = sku.find("xmin")


            xmin.text = str(int(
                (int(xmin.text) - 400) * math.cos(math.pi * rota / 180) - (int(ymin.text) - 300) * math.sin(
                    math.pi * rota / 180)) + 400)
            ymin.text = str(int(
                (int(xmin.text) - 400) * math.sin(math.pi * rota / 180) + (int(ymin.text) - 300) * math.cos(
                    math.pi * rota / 180)) + 300)

            xmax = sku.find("xmax")
            ymax = sku.find("ymax")

            xmax.text = str(int(
                (int(xmax.text) - 400) * math.cos(math.pi * rota / 180) - (int(ymax.text) - 300) * math.sin(
                    math.pi * rota / 180)) + 400)
            ymax.text = str(int(
                (int(xmax.text) - 400) * math.sin(math.pi * rota / 180) + (int(ymax.text) - 300) * math.cos(
                    math.pi * rota / 180)) + 300)

    tree.write(xml_save_path, encoding='utf-8')
    return np.array(img_new)


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
        if img_file.endswith(".jpg"):
            img_path = img_dir + "/" + img_file

            img_name = str(img_file).split(".")[0] + "_rotate5" + ".jpg"  # 图片名称
            xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
            img = image_rotate(img_path, xml_name, xml_save_name)

            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111_train"
    xml_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/data/rotate5"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/data/rotate5_annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
