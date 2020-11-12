# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : sunyihuan
# @File    : img_dir_aug_3660camera_process.py
'''
将图片处理成偏红色效果
'''

import cv2
import os
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


def img_mist(img_path):
    '''
    图片中加入水雾效果
    :param img_path: 图片地址
    :return:
    '''
    img1 = cv2.imread(img_path)  # 目标图片
    img2 = cv2.imread('C:/Users/sunyihuan/Desktop/material/lvse.jpg')  # 水雾图片
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # 统一图片大小
    # dst=cv2.add(img1,img2)
    dst = cv2.addWeighted(img1, 0.8, img2, 0.3, 0)  # 图片融合
    blank = np.zeros(img1.shape, img1.dtype)
    dst = cv2.addWeighted(dst, 0.8, blank, 0.1, 0)  # 图片融合
    return dst


def aug_bright(img, factor):
    '''
    图像亮度调整
    :param img: 图片数据
    :param factor: 亮度因子
    :return:
    '''
    img = Image.open(img)
    tmp = ImageEnhance.Brightness(img)
    img = tmp.enhance(1.1)
    enh_col = ImageEnhance.Contrast(img)
    img = enh_col.enhance(1.2)
    return np.array(img)


def img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    文件夹统一对图片做水雾或油渍处理再保存，同时保存xml文件
    :param img_dir: 原图片地址
    :param xml_dir: xml地址
    :param img_save_dir: 增强后图片保存地址
    :param xml_save_dir: 增强后xml保存地址
    :return:
    '''
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith("jpg"):
            img_path = img_dir + "/" + img_file
            img = img_mist(img_path)  # 图片加雾处理
            img_name = str(img_file).split(".")[0] + "_lv" + ".jpg"  # 图片名称
            xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
            cv2.imwrite(img_save_dir + "/" + img_name, img)
            img = aug_bright(img_save_dir + "/" + img_name, 1.2)
            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "E:/DataSets/X_3660_data/bu/20201020/JPGImages"
    xml_dir = "E:/DataSets/X_3660_data/bu/20201020/Annotations"
    img_save_dir = "E:/DataSets/X_3660_data/bu/20201020/JPGImages_aug"
    xml_save_dir = "E:/DataSets/X_3660_data/bu/20201020/Annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
