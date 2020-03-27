# -*- encoding: utf-8 -*-

"""
图片中加入白条

@File    : img_aug_white.py
@Time    : 2019/9/25 15:41
@Author  : sunyihuan
"""

from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def get_sliced(sp):
    '''
    输出白色区域像素值为255
    :param sp: 维数
    :return:
    '''
    sliced = [[175 for i in range(sp[2])] for j in range(sp[1])]
    sliced = np.array([sliced for k in range(sp[0])])
    return sliced


def aug_white(img_path):
    '''
    图片中加入白条
    :param img_path: 图片地址
    :return:
    '''
    im = Image.open(img_path)
    im = np.array(im)
    sliced = get_sliced((30, 800, 3))
    im[570:, :, :] = sliced
    sliced = get_sliced((15, 800, 3))
    im[150:165, :, :] = sliced
    im[340:355, :, :] = sliced
    im[470:485, :, :] = sliced
    img = im.astype(int)
    return img


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
        img_path = img_dir + "/" + img_file
        img = aug_white(img_path)
        img_name = str(img_file).split(".")[0] + "_white" + ".jpg"  # 图片名称
        plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
        xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
        shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "E:/DataSets/KX_FOODSets0802/JPGImages/SweetPotatoM"
    xml_dir = "E:/DataSets/KX_FOODSets0802/Anotations/SweetPotatoM"
    img_save_dir = "C:/Users/sunyihuan/Desktop/data/imgs"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/data/annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
