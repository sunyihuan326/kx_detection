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
    imgB = np.clip(imgB, 0, 255)
    imgG = np.clip(imgG, 0, 255)
    imgR = np.clip(imgR, 0, 255)

    dst[:, :, 0] = imgB
    dst[:, :, 1] = imgG
    dst[:, :, 2] = imgR
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
    for c in ["pizzafour", "pizzaone", "pizzasix", "pizzatwo", "porkchops",
              "roastedchicken"]:
        img_dir = "C:/Users/sunyihuan/Desktop/test_results_jpg/supply/{}".format(c)
        img_save_dir = "C:/Users/sunyihuan/Desktop/test_results_jpg/supply/{}_white".format(c)
        if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
        img_dir_whiteBalance(img_dir, img_save_dir)
