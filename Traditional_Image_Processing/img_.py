# -*- encoding: utf-8 -*-

"""
@File    : img_.py
@Time    : 2019/11/14 10:38
@Author  : sunyihuan
"""
from PIL import Image
import cv2
from skimage import color
import numpy as np

img_path = "C:/Users/sunyihuan/Desktop/tttttt/Cookies/2_191030X3_qkl_Cookies.jpg"


# img = cv2.imread(img_path)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
# img_hsv=np.array(img_hsv)
# print(np.array(img_hsv))
# cv2.imshow("img",img_hsv)
# cv2.waitKey(0)
def img_mist(img_path):
    '''
    图片中加入水雾效果
    :param img_path: 图片地址
    :return:
    '''
    img1 = cv2.imread(img_path)  # 目标图片
    img2 = cv2.imread('C:/Users/sunyihuan/Desktop/111.jpg')  # 水雾图片
    img2 = cv2.resize(img2, (800, 600))  # 统一图片大小
    # dst=cv2.add(img1,img2)
    dst = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)  # 图片融合
    return dst


def img_flip(img):
    img1 = cv2.imread(img_path)  # 目标图片
    img = cv2.flip(img1, 1)  #水平镜像
    return img


d = img_flip(img_path)

cv2.imshow("img", d)
cv2.waitKey(0)
