# -*- encoding: utf-8 -*-

"""
模拟可能出现的问题，图片中画直线

@File    : img_aug_drawline.py
@Time    : 2019/9/26 16:34
@Author  : sunyihuan
"""

from PIL import Image, ImageDraw
import numpy as np
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def aug_drawLine(img_path):
    '''
    图片中加入线条
    :param img_path: 图片地址
    :return:
    '''
    im = Image.open(img_path)
    draw_img = ImageDraw.Draw(im)
    for i in range(8):
        l = np.random.randint(1, 599)
        draw_img.line((0, l, 800, l), fill=(34, 139, 34), width=2)  #绿色线条
        # draw_img.line((0, l, 800, l), fill=(30, 144, 255), width=2)  # 蓝色线条
    return np.array(im)


def data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    图像增强后保存
    :param img_dir: 原图片地址
    :param xml_dir: xml地址
    :param img_save_dir: 增强后图片保存地址
    :param xml_save_dir: 增强后xml保存地址
    :return:
    '''
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith(".jpg"):
            img_path = img_dir + "/" + img_file
            img = aug_drawLine(img_path)
            img_name = str(img_file).split(".")[0] + "_line" + ".jpg"  # 图片名称
            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
            shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/JPGImages"
    xml_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/line"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/line_annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
