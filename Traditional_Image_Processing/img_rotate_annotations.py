# -*- encoding: utf-8 -*-

"""
@File    : img_rotate_annotations.py
@Time    : 2019/11/23 10:58
@Author  : sunyihuan
"""
import math
import cv2
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt


def pointRotate(img_size, srcPoint, rotate_center, rota):
    '''
    图片旋转rota角度后，点旋转角度
    :param img_size:图片尺寸，（w，h）
    :param srcPoint:目标点坐标（x，y）
    :param rotate_center:旋转中心点（x0，y0）
    :param rota:旋转角度，int
    :return:
    '''
    x1 = srcPoint[0]
    y1 = img_size[1] - srcPoint[1]

    x2 = rotate_center[0]
    y2 = img_size[1] - rotate_center[1]

    x = (x1 - x2) * math.cos(math.pi / 180.0 * rota) - (y1 - y2) * math.sin(
        math.pi / 180.0 * rota) + x2
    y = (x1 - x2) * math.sin(math.pi / 180.0 * rota) + (y1 - y2) * math.cos(
        math.pi / 180.0 * rota) + y2
    return (int(x), int(y))


def RotateBox(img_size, original_PointA, original_PointB, rota):
    '''
    检测狂矫正
    :param img_size: 图片尺寸
    :param original_PointA: 第一个点位置坐标
    :param original_PointB: 第二个点位置坐标
    :param rota: 旋转角度
    :return:
    '''
    a = pointRotate(img_size, original_PointA, (800, 600), rota)
    b = pointRotate(img_size, (original_PointB[0], original_PointA[1]), (800, 600), rota)
    c = pointRotate(img_size, (original_PointA[0], original_PointB[1]), (800, 600), rota)
    d = pointRotate(img_size, original_PointB, (800, 600), rota)
    pt1_x = int((c[0] + a[0]) / 2)
    pt1_y = int((b[1] + a[1]) / 2)
    pt2_x = int((d[0] + b[0]) / 2)
    pt2_y = int((c[1] + d[1]) / 2)
    return (pt1_x, pt1_y, pt2_x, pt2_y)


def image_rotate(img_path, xml_path):
    '''
    图片padding，将图照片自动padding成正方形
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    (w, h) = img.size
    rota = random.randint(1, 3)
    img_new = img.rotate(rota, expand=1)  # 图片旋转

    # xml标注数据修改
    tree = ET.parse(xml_path)
    root = tree.getroot()
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

            (xmin_, ymin_, xmax_, ymax_) = RotateBox((w, h), (xmin_, ymin_), (xmax_, ymax_), 10)  # 坐标框修正

            xmin.text = str(xmin_)
            ymin.text = str(ymin_)
            xmax.text = str(xmax_)
            ymax.text = str(ymax_)
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

            img_name = str(img_file).split(".")[0] + "_rotate5" + ".jpg"  # 图片名称
            xml_name = xml_dir + "/" + str(img_file).split(".")[0] + ".xml"  # xml文件名称
            xml_save_name = xml_save_dir + "/" + str(img_name).split(".")[0] + ".xml"  # xml文件保存名称
            img, tree = image_rotate(img_path, xml_name)

            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            tree.write(xml_save_name, encoding='utf-8')


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111_train"
    xml_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27_1111/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/data/rotate10"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/data/rotate10_annotations"

    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
