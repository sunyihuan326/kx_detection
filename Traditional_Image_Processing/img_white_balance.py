# -*- encoding: utf-8 -*-

"""
@File    : img_white_balance.py
@Time    : 2019/11/19 9:56
@Author  : sunyihuan
"""
import cv2
import numpy as np
import time

img_path = "C:/Users/sunyihuan/Desktop/test/1.jpg"

start_time = time.time()  # 开始时间
img = cv2.imread(img_path, 1)
cv2.imshow('111', img)
width = img.shape[1]
height = img.shape[0]
dst = np.zeros(img.shape, img.dtype)

# 1.计算三通道灰度平均值
imgB, imgG, imgR = cv2.split(img)

bAve = cv2.mean(imgB)[0]
gAve = cv2.mean(imgG)[0]
rAve = cv2.mean(imgR)[0]

Ave = (bAve + gAve + rAve) / 3

# 2计算每个通道的增益系数
# KB = (bAve + gAve + rAve) / (6 * bAve)
# KG = (bAve + gAve + rAve) / (6 * gAve)
# KR = (bAve + gAve + rAve) / (6 * rAve)
KB = Ave / bAve
KG = Ave / gAve
KR = Ave / rAve

# KB = (imgB + imgG + imgR) / (3.8 * imgB)
# KG = (imgB + imgG + imgR) / (2.2 * imgG)
# KR = (imgB + imgG + imgR) / (2.5 * imgR)

# KB = (imgB + imgG + imgR) / (2.7 * imgB)
# KG = (imgB + imgG + imgR) / (2.8 * imgG)
# KR = (imgB + imgG + imgR) / (3.5 * imgR)

# 3使用增益系数
imgB = imgB * KB  # 向下取整
imgG = imgG * KG
imgR = imgR * KR

imgB = np.clip(imgB, 0, 255)
imgG = np.clip(imgG, 0, 255)
imgR = np.clip(imgR, 0, 255)

dst[:,:,0]=imgB
dst[:,:,1]=imgG
dst[:,:,2]=imgR
# # 4将数组元素后处理
# for i in range(0, height):
#     for j in range(0, width):
#         imgb = imgB[i, j]
#         imgg = imgG[i, j]
#         imgr = imgR[i, j]
#         if imgb > 255:
#             imgb = 255
#         if imgg > 255:
#             imgg = 255
#         if imgr > 255:
#             imgr = 255
#         dst[i, j] = (imgb, imgg, imgr)
end_time = time.time()  # 结束时间
print("总耗时：", end_time - start_time)
cv2.imwrite("C:/Users/sunyihuan/Desktop/test/1_____.jpg", dst)
cv2.imshow('222', dst)
cv2.waitKey(0)
