# -*- encoding: utf-8 -*-

"""
文件夹中所有图片自动白平衡并保存

@File    : white_balance_dir.py
@Time    : 2019/12/5 10:54
@Author  : sunyihuan
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def white_balance(img_path):
    '''
    图片自动白平衡
    :param img_path: 图片地址
    :return:
    '''
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    dst = np.zeros(img.shape, img.dtype)

    # 1.计算三通道灰度平均值
    imgB, imgG, imgR = cv2.split(img)

    bAve = cv2.mean(imgB)[0]
    gAve = cv2.mean(imgG)[0]
    rAve = cv2.mean(imgR)[0]

    Ave = (bAve + gAve + rAve) / 3

    # 2.通道值调整
    KB = Ave / bAve
    KG = Ave / gAve
    KR = Ave / rAve

    # 3使用增益系数
    imgB = imgB * KB
    imgG = imgG * KG
    imgR = imgR * KR

    # # 4将数组元素后处理
    for i in range(0, height):
        for j in range(0, width):
            imgb = imgB[i, j]
            imgg = imgG[i, j]
            imgr = imgR[i, j]
            if imgb > 255:
                imgb = 255
            if imgg > 255:
                imgg = 255
            if imgr > 255:
                imgr = 255
            dst[i, j] = (imgb, imgg, imgr)
    return dst


def img_dir_whiteBalance(img_dir, img_save_dir):
    '''
    整个文件夹下图片白平衡
    :param img_dir: 原图片地址
    :param img_save_dir: 平衡后图片保存地址
    :return:
    '''
    for img_file in tqdm(os.listdir(img_dir)):
        img_path = img_dir + "/" + img_file
        img = white_balance(img_path)
        img_name = str(img_file).split(".")[0] + "_whiteBalance" + ".jpg"  # 图片名称
        cv2.imwrite(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/rgb_test/PorkChops_new/top"
    img_save_dir = "C:/Users/sunyihuan/Desktop/rgb_test/PorkChops_top_white"
    img_dir_whiteBalance(img_dir, img_save_dir)
