# -*- encoding: utf-8 -*-

"""
模型训练时数据增强方法效果查看
说明：文件夹中所有图片按增强方法处理后保存，并查看效果

@File    : data_aug_check.py
@Time    : 2019/12/30 15:39
@Author  : sunyihuan
"""

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import random
from skimage import util
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def data_aug(image_path):
    '''
    单张图片增强处理
    :param image_path: 图片地址
    :return:
    '''
    image = Image.open(image_path)  # 读取图片
    # image = ImageEnhance.Contrast(image)  # 对比度增强
    # image = image.enhance(1.2)  # 增强系数[0.6, 1.2]
    # image = ImageEnhance.Brightness(image)  # 亮度调整
    # image = image.enhance(1.2)  # 亮度调整系数[0.7, 1.2]
    # image = ImageEnhance.Sharpness(image)  # 锐度增强
    # image = image.enhance(2)  # 亮度调整系数[0.5, 2]
    image = ImageEnhance.Color(image)  # 颜色增强
    image = image.enhance(2)
    # image = ImageOps.autocontrast(image, 5)
    # image = util.random_noise(np.array(image), mode="gaussian")  # 加入高斯噪声,输出值为[0,1],需乘以255
    # image = image * 255

    return np.array(image).astype(int)


def dir_aug(JPGImages, AugJPGImages):
    '''
    文件夹下所有图片增强处理
    :param JPGImages: 原图片文件地址
    :param AugJPGImages: 增强后图片保存地址
    :return:
    '''
    images_list = os.listdir(JPGImages)
    for images in tqdm(images_list):
        if images.endswith(".jpg"):
            image = data_aug(JPGImages + "/" + images)
            plt.imsave(AugJPGImages + "/" + images, image.astype(np.uint8))  # 保存图片


if __name__ == "__main__":
    JPGImages = "C:/Users/sunyihuan/Desktop/aug_data_test/JPGImages"
    AugJPGImages = "C:/Users/sunyihuan/Desktop/aug_data_test/AugJPGImages"
    dir_aug(JPGImages, AugJPGImages)
