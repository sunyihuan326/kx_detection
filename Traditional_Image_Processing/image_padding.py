# -*- encoding: utf-8 -*-

"""
图片padding成正方形，上下用黑色填充，并且重新修改和保存xml数据

@File    : image_padding.py
@Time    : 2019/10/8 11:05
@Author  : sunyihuan
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os


def image_padding(img_path):
    '''
    图片padding，将图照片自动padding成正方形
    :param img_path: 图片地址
    :return:
    '''
    image = Image.open(img_path)
    (w, h) = image.size
    image = np.array(image)

    channel_one = image[:, :, 0]
    channel_two = image[:, :, 1]
    channel_three = image[:, :, 2]

    if w > h:
        pad_d = int((w - h) / 2)
        channel_one = np.pad(channel_one, ((pad_d, pad_d), (0, 0)), 'constant', constant_values=(0, 0))
        channel_two = np.pad(channel_two, ((pad_d, pad_d), (0, 0)), 'constant', constant_values=(0, 0))
        channel_three = np.pad(channel_three, ((pad_d, pad_d), (0, 0)), 'constant', constant_values=(0, 0))
        image = np.dstack((channel_one, channel_two, channel_three))
    elif w < h:
        pad_d = int((h - w) / 2)
        channel_one = np.pad(channel_one, ((0, 0), (pad_d, pad_d)), 'constant', constant_values=(0, 0))
        channel_two = np.pad(channel_two, ((0, 0), (pad_d, pad_d)), 'constant', constant_values=(0, 0))
        channel_three = np.pad(channel_three, ((0, 0), (pad_d, pad_d)), 'constant', constant_values=(0, 0))
        image = np.dstack((channel_one, channel_two, channel_three))
    else:
        image = image
    return image, w, h


def xml_padding(xml_path, w, h, xml_save_path):
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
            if w > h:
                ymin = sku.find("ymin")
                ymax = sku.find("ymax")
                ymin.text = str(int(ymin.text) + int((w - h) / 2))
                ymax.text = str(int(ymax.text) + int((w - h) / 2))
            elif h > w:
                xmin = sku.find("xmin")
                xmax = sku.find("xmax")
                xmin.text = str(int(xmin.text) + int((h - w) / 2))
                xmax.text = str(int(xmax.text) + int((h - w) / 2))
            else:
                print("no change")

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
        img_path = img_dir + "/" + img_file
        img, w, h = image_padding(img_path)  # 图片padding处理

        img_name = str(img_file).split(".")[0] + "_pad" + ".jpg"  # 图片名称
        plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片

        xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
        xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
        xml_padding(xml_name, w, h, xml_save_name)  # xml文件标注更改、保存


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920/JPGImages"
    xml_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920/Annotations"
    img_save_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920_padding/JPGImages"
    xml_save_dir = "E:/DataSets/KX_FOODSets_model_data/26classes_0920_padding/Annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)

