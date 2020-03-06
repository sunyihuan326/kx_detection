# -*- encoding: utf-8 -*-

"""
滤波、加入噪声、亮度调整、对比度增强
文件夹中统一图像增广，并保存到单独文件夹中，包含xml文件、

@File    : img_dir_aug.py
@Time    : 2019/9/25 10:26
@Author  : sunyihuan
"""
from PIL import Image, ImageEnhance, ImageFilter
from skimage import util
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm


class aug(object):
    '''
    图像增强
    '''

    def aug_filter(self, img, mode):
        '''
        图像滤波
        :param img: 图片数据
        :param mode: 滤波方法
        :return:
        '''
        img = Image.open(img)
        if mode == "BLUR":
            img = img.filter(ImageFilter.BLUR)
        elif mode == "EDGE_ENHANCE":
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        elif mode == "SHARPEN":
            img = img.filter(ImageFilter.SHARPEN)
        elif mode == "DETAIL":
            img = img.filter(ImageFilter.DETAIL)
        else:
            print("Input filter method!!!")

        return np.array(img)

    def aug_contrast(self, img, factor):
        '''
        图像对比度增强
        :param img: 图像数据
        :param factor: 对比度因子
        :return:
        '''
        img = Image.open(img)
        enh_con = ImageEnhance.Contrast(img)
        img = enh_con.enhance(factor)
        return np.array(img)

    def aug_bright(self, img, factor):
        '''
        图像亮度调整
        :param img: 图片数据
        :param factor: 亮度因子
        :return:
        '''
        img = Image.open(img)
        tmp = ImageEnhance.Brightness(img)
        img = tmp.enhance(factor)
        return np.array(img)

    def aug_noise(self, img, mode):
        '''
        图像加入噪声
        :param img: 图片数据
        :param mode: 噪声类别，   含：gaussian、salt、localvar、poisson、pepper、s&p、speckle
        :return:
        '''
        img = Image.open(img)
        img = np.array(img)
        img = util.random_noise(img, mode="gaussian")  # 加入高斯噪声,输出值为[0,1],需乘以255
        img = img * 255
        return img.astype(int)


def data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    图像增强后保存
    :param img_dir: 原图片地址
    :param xml_dir: xml地址
    :param img_save_dir: 增强后图片保存地址
    :param xml_save_dir: 增强后xml保存地址
    :return:
    '''
    au = aug()
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith("jpg"):
            img = img_dir + "/" + img_file
            mode = "gaussian"
            # img = au.aug_filter(img, mode=mode)  # 滤波
            img = au.aug_noise(img, mode=mode)  # 加入噪声
            # img = au.aug_bright(img, 1.2)  # 亮度调整
            # img = au.aug_contrast(img, 1.2)  #对比度增强
            img_name = str(img_file).split(".")[0] + "_" + mode + ".jpg"  # 图片名称
            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
            shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/JPGImages"
    xml_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/gaussian"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/gaussian_annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
