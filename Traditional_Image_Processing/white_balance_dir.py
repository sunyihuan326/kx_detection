# -*- encoding: utf-8 -*-

"""
文件夹中所有图片自动白平衡并保存

@File    : white_balance_dir.py
@Time    : 2019/12/5 10:54
@Author  : sunyihuan
"""
import cv2
import numpy as np
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

    KB = 1.03
    KG = 1.03
    KR = 0.9
    # 3使用增益系数
    imgB = imgB * KB  # 向下取整
    imgG = imgG * KG
    imgR = imgR * KR

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
    img_dir = "C:/Users/sunyihuan/Desktop/X3_2phase"
    img_save_dir = "C:/Users/sunyihuan/Desktop/X3_2phase_white_balance"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    img_dir_whiteBalance(img_dir, img_save_dir)
    # for c in ["beefsteak", "cartooncookies", "chickenwings", "chiffoncake6", "chiffoncake8",
    #           "cookies", "cranberrycookies", "cupcake", "eggtart", "peanuts",
    #           "pizzacut", "pizzaone", "pizzatwo", "porkchops", "potatocut",
    #           "potatol", "potatos", "sweetpotatocut", "sweetpotatol", "sweetpotatos",
    #           "roastedchicken", "toast", ]:
    #     img_dir = "E:/WLS_originalData/3660camera_data202007/X3_original/{}".format(c)
    #     img_save_dir = "E:/WLS_originalData/3660camera_data202007/X3_white_balance1/{}_white".format(c)
    #     if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    #     img_dir_whiteBalance(img_dir, img_save_dir)
