# -*- encoding: utf-8 -*-

"""
@File    : img_white_balance.py
@Time    : 2019/11/19 9:56
@Author  : sunyihuan
"""
import cv2
import numpy as np

img_path = "C:/Users/sunyihuan/Desktop/test/ceshi.jpg"
img = cv2.imread(img_path, 1)
cv2.imshow('111', img)
width = img.shape[1]
height = img.shape[0]
dst = np.zeros(img.shape, img.dtype)

# 1.计算三通道灰度平均值
imgB = img[:, :, 0]
imgG = img[:, :, 1]
imgR = img[:, :, 2]
bAve = cv2.mean(imgB)[0]
gAve = cv2.mean(imgG)[0]
rAve = cv2.mean(imgR)[0]
aveGray = (int)(bAve + gAve + rAve) / 3

# 2计算每个通道的增益系数
bCoef = aveGray / bAve
gCoef = aveGray / gAve
rCoef = aveGray / rAve

# 3使用增益系数
imgB = np.floor((imgB * bCoef))  # 向下取整
imgG = np.floor((imgG * gCoef))
imgR = np.floor((imgR * rCoef))

# 4将数组元素后处理
maxB = np.max(imgB)
minB = np.min(imgB)
maxG = np.max(imgG)
minG = np.min(imgG)
maxR = np.max(imgR)
minR = np.min(imgR)
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
cv2.imshow('222', dst)
cv2.waitKey(0)
