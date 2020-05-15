# -*- coding: utf-8 -*-
# @Time    : 2020/3/19
# @Author  : sunyihuan
# @File    : resize_aug_dir.py

'''
resize图片，并同时修改xml中标注数据
'''
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt


def image_xml_resize(img_path, xml_path):
    '''
    图片resize,并同时修改xml数据
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    (img_w, img_h) = img.size

    w = random.randrange(800,1600)
    h = int((w / 800) * 600)
    img_new = img.resize((w, h), Image.ANTIALIAS)  # 图片尺寸变化

    # xml标注数据修改
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for s in root.findall("size"):  # 修改宽、高数据
        width = s.find("width")
        width.text = str(w)
        height = s.find("height")
        height.text = str(h)
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            ymin = sku.find("ymin")
            xmin = sku.find("xmin")
            xmax = sku.find("xmax")
            ymax = sku.find("ymax")
            xmin_ = int(xmin.text)
            ymin_ = int(ymin.text)
            xmax_ = int(xmax.text)
            ymax_ = int(ymax.text)

            # 坐标框修正
            xmin.text = str(int(float(xmin_ * w) / img_w))
            ymin.text = str(int(float(ymin_ * h) / img_h))
            xmax.text = str(int(float(xmax_ * w) / img_w))
            ymax.text = str(int(float(ymax_ * h) / img_h))
    return np.array(img_new), tree


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

            img_name = str(img_file).split(".")[0] + "_resize_l" + ".jpg"  # 图片名称
            xml_name = xml_dir + "/" + str(img_file).split(".jpg")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".jpg")[0] + ".xml"  # xml文件保存名称
            img, tree = image_xml_resize(img_path, xml_name)

            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            tree.write(xml_save_name, encoding='utf-8')


if __name__ == "__main__":
    img_dir = "E:/DataSets/KXDataAll/JPGImages"
    xml_dir = "E:/DataSets/KXDataAll/Annotations"
    img_save_dir = "E:/DataSets/KXDataAll/JPGImages_resize_l"
    xml_save_dir = "E:/DataSets/KXDataAll/Annotations_resize_l"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
