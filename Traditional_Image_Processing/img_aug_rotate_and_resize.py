# -*- encoding: utf-8 -*-

"""
图片旋转和压缩尺寸

@File    : img_aug_rotate.py
@Time    : 2019/11/8 11:45
@Author  : sunyihuan
"""
import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm


def pic_rotate(img_path):
    '''
    图片旋转1-5度
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    rota = random.randint(1, 5)
    img_new = img.rotate(rota, expand=1)
    return img_new


def pic_crop(img_path):
    '''
    图片区域裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    w, h = img.size
    img_new = img.crop((100, 0, w, h))
    return img_new


def pic_resize(img_path):
    '''
    图片拉伸后裁剪
    :param img_path: 图片地址
    :return:
    '''
    img = Image.open(img_path)
    img_new = img.resize((800, 800))
    img_new = img_new.crop((0, 100, 800, 700))
    return img_new


if __name__ == "__main__":
    # img_path = "C:/Users/sunyihuan/Desktop/tttttt/p/113_191011size6_Pizzaone.jpg"
    img_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27classes1025_test"
    img_save_dir = "E:/DataSets/KX_FOODSets_model_data/X_KX_data_27classes1025_test_crop"
    for img in tqdm(os.listdir(img_dir)):
        img_path = img_dir + "/" + img
        img_new = pic_crop(img_path)
        img_new.save(img_save_dir + "/" + img.split(".")[0] + "_crop" + ".jpg")
