# -*- encoding: utf-8 -*-

"""
边缘检测

@File    : detection.py
@Time    : 2019/10/17 13:43
@Author  : sunyihuan
"""

# import cv2
img_path="C:/Users/sunyihuan/Desktop/test_detection/1_191017X1_Cookies.jpg"
# def box_detect():
#     img = cv2.imread('C:/Users/sunyihuan/Desktop/test_detection/1_191017X1_Cookies.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, binary = cv2.threshold(gray, 100, 150, cv2.THRESH_BINARY)  # 灰度阈值
#
#     # 对binary去噪，腐蚀与膨胀
#     binary = cv2.erode(binary, None, iterations=2)
#     binary = cv2.dilate(binary, None, iterations=2)
#     cv2.imwrite('temp_img/binary.jpg', binary)
#
#     # contours是轮廓本身，hierarchy是每条轮廓对应的属性。
#     # cv2.RETR_TREE建立一个等级树结构的轮廓。cv2.CHAIN_APPROX_SIMPLE矩形轮廓只需4个点来保存轮廓信息
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     min_num = 250000
#     max_num = 2700 * 4000
#     for contour in contours[1:]:
#
#         x, y, w, h = cv2.boundingRect(contour)  # 外接矩形
#
#         if (w * h) > min_num:
#             if (w * h) < max_num:
#                 out = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.imwrite('temp_img/1-box.jpg', out)
#                 print('find box')

import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建新的图像black
black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)


for cnt in contours:
    # 轮廓周长也被称为弧长。可以使用函数 cv2.arcLength() 计算得到。这个函数的第二参数可以用来指定对象的形状是闭合的（True） ，还是打开的（一条曲线）
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    # 函数approxPolyDP来对指定的点集进行逼近，cnt是图像轮廓，epsilon表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离。
    # 第三个函数参数若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # convexHull检查一个曲线的凸性缺陷并进行修正，参数cnt是图像轮廓。
    hull = cv2.convexHull(cnt)
    # 勾画图像原始的轮廓
    cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # 用多边形勾画轮廓区域
    cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    # 修正凸性缺陷的轮廓区域
    cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
# 显示图像
cv2.imshow("hull", black)
cv2.waitKey()
cv2.destroyAllWindows()
#
# if __name__=="__main__":
#     box_detect()