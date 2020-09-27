# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : sunyihuan
# @File    : img_mist.py
'''
图片融合
'''

import cv2
import os
from tqdm import tqdm
import shutil
import numpy as np


def img_mist(img_path):
    '''
    图片中加入水雾效果
    :param img_path: 图片地址
    :return:
    '''
    img1 = cv2.imread(img_path)  # 目标图片
    img2 = cv2.imread('C:/Users/sunyihuan/Desktop/material/nuanhuang.jpg')  # 水雾图片
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # 统一图片大小
    # dst=cv2.add(img1,img2)
    dst = cv2.addWeighted(img1, 0.8, img2, 0.3, 0)  # 图片融合
    blank = np.zeros(img1.shape, img1.dtype)
    dst = cv2.addWeighted(dst, 0.8, blank, 0.2, 0)  # 图片融合
    return dst


if __name__ == "__main__":
    img_path = "C:/Users/sunyihuan/Desktop/test_img/0_kaopan_beefsteak.jpg"
    mis = img_mist(img_path)
    cv2.imshow('222', mis)
    img_path_save = "C:/Users/sunyihuan/Desktop/test_img/0_kaopan_beefsteak_aug.jpg"
    cv2.imwrite("C:/Users/sunyihuan/Desktop/test_img/0_kaopan_beefsteak_aug.jpg", mis)
    # cv2.waitKey(0)

    from skimage import data, exposure, img_as_float
    import matplotlib.pyplot as plt
    from PIL import ImageEnhance, Image

    img = Image.open(img_path_save)
    enh_bri = ImageEnhance.Brightness(img)
    brightness = 1.2
    image_brightened1 = enh_bri.enhance(brightness)
    enh_con = ImageEnhance.Contrast(image_brightened1)
    image_brightened1 = enh_con.enhance(1.1)
    image_brightened1.show()
    image_brightened1.save("C:/Users/sunyihuan/Desktop/test_img/0_kaopan_beefsteak_aug.jpg")
