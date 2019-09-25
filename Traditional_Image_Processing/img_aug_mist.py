# -*- encoding: utf-8 -*-

"""
图片统一加入水雾或油渍，再保存

@File    : img_aug_mist.py
@Time    : 2019/9/25 15:11
@Author  : sunyihuan
"""

import cv2
import os
from tqdm import tqdm
import shutil


def img_mist(img_path):
    '''
    图片中加入水雾效果
    :param img_path: 图片地址
    :return:
    '''
    img1 = cv2.imread(img_path)  # 目标图片
    img2 = cv2.imread('C:/Users/sunyihuan/Desktop/shui.jpg')  # 水雾图片
    img2 = cv2.resize(img2, (800, 600))  # 统一图片大小
    # dst=cv2.add(img1,img2)
    dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)  # 图片融合
    return dst


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
        img_path = img_dir + "/" + img_file
        img = img_mist(img_path)  # 图片加雾处理
        img_name = str(img_file).split(".")[0] + "_mist" + ".jpg"  # 图片名称
        xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
        cv2.imwrite(img_save_dir + "/" + img_name, img)
        shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets0802/JPGImages/SweetPotatoM"
    xml_dir = "E:/DataSets/KX_FOODSets0802/Anotations/SweetPotatoM"
    img_save_dir = "C:/Users/sunyihuan/Desktop/data/imgs"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/data/annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    img_dir_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
