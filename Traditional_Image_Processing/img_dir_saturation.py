# -*- encoding: utf-8 -*-

"""
@File    : img_dir_saturation.py
@Time    : 2019/11/26 15:38
@Author  : sunyihuan
"""
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


class aug(object):
    '''
    图像增强
    '''

    def PSAlgorithm(self, rgb_img, increment):
        '''
        图片饱和度调整
        :param rgb_img: 图片
        :param increment: 增量,取值为[-1,1]
        :return:
        '''
        img = rgb_img * 1.0
        img_min = img.min(axis=2)
        img_max = img.max(axis=2)
        img_out = img

        # 获取HSL空间的饱和度和亮度
        delta = (img_max - img_min) / 255.0
        value = (img_max + img_min) / 255.0
        L = value / 2.0

        # s = L<0.5 ? s1 : s2
        mask_1 = L < 0.5
        s1 = delta / (value)
        s2 = delta / (2 - value)
        s = s1 * mask_1 + s2 * (1 - mask_1)

        # 增量大于0，饱和度指数增强
        if increment >= 0:
            # alpha = increment+s > 1 ? alpha_1 : alpha_2
            temp = increment + s
            mask_2 = temp > 1
            alpha_1 = s
            alpha_2 = s * 0 + 1 - increment
            alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)

            alpha = 1 / alpha - 1
            img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
            img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
            img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

        # 增量小于0，饱和度线性衰减
        else:
            alpha = increment
            img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
            img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
            img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

        img_out = img_out / 255.0

        # RGB颜色上下限处理(小于0取0，大于1取1)
        mask_3 = img_out < 0
        mask_4 = img_out > 1
        img_out = img_out * (1 - mask_3)
        img_out = img_out * (1 - mask_4) + mask_4

        return img_out * 255


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
    increment = 0.3  # 范围-1到1
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith("jpg"):
            img = img_dir + "/" + img_file
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = au.PSAlgorithm(img, increment)  # 加入噪声
            img_name = str(img_file).split(".")[0] + "_" + "saturation3" + ".jpg"  # 图片名称
            plt.imsave(img_save_dir + "/" + img_name, img.astype(np.uint8))  # 保存图片
            xml_name = str(img_name).split(".")[0] + ".xml"  # xml文件名称
            shutil.copy(xml_dir + "/" + str(img_file).split(".")[0] + ".xml", xml_save_dir + "/" + xml_name)  # 拷贝xml数据


if __name__ == "__main__":
    img_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/JPGImages"
    xml_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/train/Annotations"
    img_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/saturation3"
    xml_save_dir = "C:/Users/sunyihuan/Desktop/peanuts_all/saturation3_annotations"
    if not os.path.exists(img_save_dir): os.mkdir(img_save_dir)
    if not os.path.exists(xml_save_dir): os.mkdir(xml_save_dir)
    data_aug(img_dir, xml_dir, img_save_dir, xml_save_dir)
