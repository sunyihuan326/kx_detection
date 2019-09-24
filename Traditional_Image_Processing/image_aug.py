# -*- encoding: utf-8 -*-

"""
图像增广方法
@File    : image_aug.py
@Time    : 2019/9/24 16:29
@Author  : sunyihuan
"""
from PIL import Image, ImageFilter
from skimage import util
import numpy as np
import matplotlib.pyplot as plt

img_path = "C:/Users/sunyihuan/Desktop/test/3.jpg"

im = Image.open(img_path)
# blurf = im.filter(ImageFilter.BLUR)  #模糊滤波
blurf = im.filter(ImageFilter.EDGE_ENHANCE)  # 边界增强滤波

# blurf.show()

im = np.array(im)
noise_gs_img = util.random_noise(im, mode="gaussian")  #加入高斯噪声
# plt.figure()
plt.imshow(noise_gs_img)
plt.show()


