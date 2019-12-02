# -*- encoding: utf-8 -*-

"""
@File    : img_aug_rotate.py
@Time    : 2019/11/8 11:45
@Author  : sunyihuan
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def pic_rotate(img_path):
    '''
    图片旋转1-5度
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    rota = random.randint(1, 5)
    img_new = img.rotate(rota, expand=1)
    return img_new


def pic_crop(img_path):
    '''
    图片区域裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    w, h = img.size
    img_new = img.crop((100, 0, w, h))
    return img_new


def pic_resize(img_path, xml_path):
    '''
    图片拉伸后裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    w, h = img.size  # 原图尺寸
    resize_wh = (800, 800)  # resize尺寸
    img_new = img.resize(resize_wh)  # 图片resize
    crop_d = (0, 100, 800, 700)  # 裁剪位置
    img_new = img_new.crop(crop_d)  # 图片裁剪

    # xml文件中对应坐标调整
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object1 in root.findall('object'):
        for sku in object1.findall('bndbox'):
            ymin = sku.find("ymin")
            ymax = sku.find("ymax")
            xmin = sku.find("xmin")
            xmax = sku.find("xmax")
            # resize坐标调整
            xmin.text = str(int(int(xmin.text) * resize_wh[0] / w))
            xmax.text = str(int(int(xmax.text) * resize_wh[0] / w))
            ymin.text = str(int(int(ymin.text) * resize_wh[1] / h))
            ymax.text = str(int(int(ymax.text) * resize_wh[1] / h))

            # 裁剪坐标调整
            xmin.text = str(max(int(xmin.text) - crop_d[0], 0))
            xmax.text = str(min(int(xmax.text) - crop_d[0], 800))
            ymin.text = str(max(int(ymin.text) - crop_d[1], 0))
            ymax.text = str(min(int(ymax.text) - crop_d[1], 600))

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
        if img_file.endswith("jpg"):
            img_path = img_dir + "/" + img_file

            img_name = str(img_file).split(".")[0] + "_cropy" + ".jpg"  # 图片名称
            xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
            img, tree = pic_resize(img_path, xml_name)

            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            tree.write(xml_save_name, encoding='utf-8')  # xml文件写入


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/JPGImages"
    xml_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/cropy"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/cropy_annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
