# -*- encoding: utf-8 -*-

"""
图片色度调整

@File    : img_aug_saturation.py
@Time    : 2019/11/26 15:27
@Author  : sunyihuan
"""
import cv2
import sys
import matplotlib.pyplot as plt

def PSAlgorithm(rgb_img, increment):
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

    return img_out


if __name__ == "__main__":
    path = 'C:/Users/sunyihuan/Desktop/test_img/1_20200730_X1_bottom_kaojia_chiffoncake8.jpg'
    increment = 0.2  # 范围-1到1

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_new = PSAlgorithm(img, increment)

    plt.figure("img_original")
    plt.imshow(img / 255.0)
    plt.axis('off')

    plt.figure("img_saturate")
    plt.imshow(img_new)
    plt.axis('off')

    plt.show()
