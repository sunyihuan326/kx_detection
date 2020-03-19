# -*- encoding: utf-8 -*-

"""
抠图

@File    : tt.py
@Time    : 2019/10/17 13:16
@Author  : sunyihuan
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('E:/已标数据备份/二期数据/test/JPGImages/1_200304_X5__kaojia(bupuxizhi)_shang_cornOne.jpg')
print(img)
# mask = np.zeros(img.shape[:2], np.uint8)
#
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
#
# rect = (20, 20, 413, 591)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
# img += 255 * (1 - cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
# # plt.imshow(img)
# # plt.show()
# img = np.array(img)
# mean = np.mean(img)
# img = img - mean
# img = img * 0.9 + mean * 0.9
# img /= 255
# plt.imshow(img)
# plt.show()