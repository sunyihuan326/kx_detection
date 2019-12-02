# -*- encoding: utf-8 -*-

"""
图像增广方法
@File    : image_aug.py
@Time    : 2019/9/24 16:29
@Author  : sunyihuan
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from skimage import util
import numpy as np
import matplotlib.pyplot as plt

img_path = "C:/Users/sunyihuan/Desktop/test/3.jpg"

im = Image.open(img_path)

# 图片中画线
# draw_img = ImageDraw.Draw(im)
# draw_img.line((0, 400, 800, 400), fill=(34, 139, 34))
# im.show()
# blurf = im.filter(ImageFilter.BLUR)  #模糊滤波
# blurf = im.filter(ImageFilter.EDGE_ENHANCE)  # 边界增强滤波，其他滤波：
#  SHARPEN、MaxFilter、MedianFilter、SMOOTH_MORE、EDGE_ENHANCE_MORE等
# blurf.show()


# 调整图片对比度
# enh_con = ImageEnhance.Contrast(im)
# contrast = 1.6
# img_contrasted = enh_con.enhance(contrast)
# img_contrasted.show()

# 调整亮度
# tmp = ImageEnhance.Brightness(im)
# img_bright = tmp.enhance(1.5)
# img_bright.show()

im = np.array(im)
print(im)
# noise_gs_img = util.random_noise(im, mode="gaussian")  # 加入高斯噪声
noise_gs_img = util.random_noise(im, mode="speckle")  #加入椒盐噪声，其他噪声方法：localvar、poisson、pepper、s&p、speckle
print(noise_gs_img*255)
plt.figure()
plt.imshow(noise_gs_img)
plt.show()

# 底部白条
# ones = np.ones((600, 800, 3))
# sliced = [[255 for i in range(3)] for j in range(800)]
# sliced = np.array([sliced for k in range(30)])
# im[570:, :, :] = sliced
# img = im.astype(int)
#
#
# plt.figure()
# plt.imshow(img)
# plt.show()


# 图片中加入水雾效果
# import cv2
# img1 = cv2.imread(img_path)
# img2 = cv2.imread('C:/Users/sunyihuan/Desktop/wuzi1.jpg')
# img2 = cv2.resize(img2, (800, 600))  # 统一图片大小
# # dst=cv2.add(img1,img2)
# dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
#
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
