# -*- coding: utf-8 -*-
# @Time    : 2021/2/2
# @Author  : sunyihuan
# @File    : img_seg_thresh.py
'''
基于阈值方法图像分割
'''


import cv2

img = cv2.imread('output.jpg', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # binary （黑白二值）
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # （黑白二值反转）
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)  # 得到的图像为多像素值
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # 高于阈值时像素设置为255，低于阈值时不作处理
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)  # 低于阈值时设置为255，高于阈值时不作处理

print(ret)
th6= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
th7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th8 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 简单滤波
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波

cv2.imshow('thresh1', thresh1)
cv2.imshow('thresh2', thresh2)
cv2.imshow('thresh3', thresh3)
cv2.imshow('thresh4', thresh4)
cv2.imshow('thresh5', thresh5)
cv2.imshow('thresh6', th6)
cv2.imshow('thresh4', th7)
cv2.imshow('thresh5', th8)

cv2.imshow('th1', th1)
cv2.imshow('th2', th2)

# cv2.imshow('grey-map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()